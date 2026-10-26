# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Interactive REPL and CLI for Levanter model inference.

Allows loading/unloading models and submitting chat/text completion requests.

Interactive REPL Usage:
    uv run src/levanter/main/inference_repl.py
    uv run src/levanter/main/inference_repl.py --checkpoint /path/to/checkpoint

CLI Usage:
    uv run src/levanter/main/inference_repl.py --command=complete --args="The chicken liked to eat" --checkpoint=meta-llama/Llama-3.2-1B-Instruct
"""

import asyncio
import json
import logging
import os
import shlex
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, Optional

import jax
import jax.random as jrandom
import jmp
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel

import levanter
import levanter.config
from levanter.quantization import cast_model_preserving_quantization_state
from levanter.compat.hf_checkpoints import _is_hf_model_id
from levanter.main.model_init import ModelInitContext, load_model_from_source, prepare_model_init_context
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.openai import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    InferenceServer,
    InferenceServerConfig,
    _create_chat_completion,
    _create_completion,
)
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

logger = logging.getLogger(__name__)
console = Console()

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


def _load_model(
    trainer_config: TrainerConfig,
    model_config: LmConfig,
    hf_checkpoint: str | None,
    levanter_checkpoint: str | None,
    *,
    tokenizer: MarinTokenizer,
    key,
) -> LmHeadModel:
    """Load inference weights with the shared source and computation-cast contract."""
    if levanter_checkpoint is None and hf_checkpoint is None:
        raise ValueError("Must specify either checkpoint_path or hf_checkpoint")
    if levanter_checkpoint is not None and hf_checkpoint is not None:
        raise ValueError("Specify only one of checkpoint_path or hf_checkpoint")

    if hf_checkpoint is not None:
        context = prepare_model_init_context(
            model_config,
            tokenizer=tokenizer,
            initialize_from_hf=hf_checkpoint,
            use_hf_model_config=True,
        )
    else:
        context = ModelInitContext(model=model_config, converter=None, use_hf_model_config=False)
    with trainer_config.use_device_mesh():
        Vocab = round_axis_for_partitioning(Axis("vocab", len(tokenizer)), trainer_config.compute_axis_mapping)
        return load_model_from_source(
            context=context,
            Vocab=Vocab,
            model_key=key,
            parameter_axis_mapping=trainer_config.parameter_axis_mapping,
            compute_axis_mapping=trainer_config.compute_axis_mapping,
            batch_axis_name=trainer_config.batch_axis_name,
            load_dtype=trainer_config.mp.compute_dtype,
            cast_to_param=lambda model: cast_model_preserving_quantization_state(
                model, trainer_config.mp.cast_to_compute
            ),
            hf_ref=hf_checkpoint if hf_checkpoint is not None else False,
            checkpoint_path=levanter_checkpoint,
        )


@dataclass
class InferenceReplConfig:
    """Configuration for the inference REPL."""

    # Model and training configuration
    checkpoint: str

    model: LmConfig
    tokenizer: str | None = None

    trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(
            mp=jmp.get_policy("p=f32,c=f32"),
            mesh=MeshConfig(
                axes={"model": 1},
                shared_mapping={"mlp": "model", "heads": "model"},
            ),
        )
    )
    server: InferenceServerConfig = field(
        default_factory=lambda: InferenceServerConfig(
            service=InferenceEngineConfig(
                page_size=8,
                max_pages=64 * 8,
                max_seq_len=64,
                max_seqs=2,
                max_queued_tokens=64,
                max_seqs_in_prefill=1,
                max_prefill_size=64,
                max_rounds=4,
            ),
        )
    )

    # Generation parameters
    temperature: float = 0.7
    seed: int = 42
    max_tokens: int = 64

    # CLI mode parameters
    command: Optional[str] = None
    args: str = ""


class ReplContext:
    """Command handler for both REPL and CLI modes."""

    server: InferenceServer | None
    model_name: Optional[str]
    config: InferenceReplConfig

    def __init__(self, config: InferenceReplConfig):
        trainer = config.trainer
        if config.model.requires_explicit_mesh_axes:
            trainer = replace(trainer, use_explicit_mesh_axes=True)
        self.config = replace(
            config,
            trainer=trainer,
            server=replace(
                config.server,
                trainer=trainer,
                service=replace(config.server.service, compute_dtype=trainer.mp.compute_dtype),
            ),
        )
        self.server = None
        self.model_name = None

        self.commands: Dict[str, Callable] = {
            "load": self.load,
            "unload": self.unload,
            "chat": self.chat,
            "complete": self.complete,
            "batch": self.batch,
            "help": self.show_help,
            "serve": self.serve,
        }

    def serve(self):
        if not self.server:
            console.print("[red]No model loaded. Use 'load' command first[/red]")
            return

        console.print("[blue]Starting inference server...[/blue]")
        self.server.serve()

    def execute(self, cmd_name: str, *args, **kwargs):
        """Execute a command by name."""
        if cmd_name in self.commands:
            return self.commands[cmd_name](*args, **kwargs)
        else:
            console.print(f"[red]Unknown command: {cmd_name}[/red]")

    def load(self, path: str, tokenizer: Optional[str] = None, **kwargs):
        """Load a model from checkpoint or HuggingFace."""
        console.print(f"[blue]Loading {path}...[/blue]")

        is_hf_model = (Path(path) / "config.json").is_file() or _is_hf_model_id(path)
        tokenizer_path = tokenizer or self.config.tokenizer or (path if is_hf_model else None)
        if tokenizer_path is None:
            raise ValueError("Must specify a tokenizer for Levanter checkpoints")
        loaded_tokenizer = load_tokenizer(tokenizer_path)
        model = _load_model(
            trainer_config=self.config.trainer,
            model_config=self.config.model,
            hf_checkpoint=path if is_hf_model else None,
            levanter_checkpoint=None if is_hf_model else path,
            tokenizer=loaded_tokenizer,
            key=jrandom.PRNGKey(self.config.server.seed),
        )

        if self.server is not None:

            def _reload(_current_model: LmHeadModel) -> LmHeadModel:
                return model

            self.server.reload(_reload)
        else:
            with self.config.trainer.use_device_mesh():
                self.server = InferenceServer.create(self.config.server, model=model, tokenizer=loaded_tokenizer)

        console.print(f"[green]✓ Loaded {path}[/green]")

    def unload(self):
        """Unload the current model."""
        if self.server:
            console.print(f"[blue]Unloading {self.model_name}...[/blue]")
            self.server.shutdown()
            self.server = None
            console.print("[green]✓ Model unloaded[/green]")
        else:
            console.print("[yellow]No model loaded[/yellow]")

    def chat(self, message: Optional[str] = None):
        """Chat with the model."""
        if not self.server:
            console.print("[red]No model loaded. Use 'load' command first[/red]")
            return

        if message:
            messages = [ChatMessage(role="user", content=message)]
            self._run_chat_completion(messages)
        else:
            self._run_chat_session()

    def show_help(self):
        """Show help text."""
        help_text = """
[bold cyan]Commands:[/bold cyan]
  load <path|hf:model>      Load model (e.g., load meta-llama/Llama-3.2-1B)
  unload                    Unload current model
  chat [text]               Chat with model (interactive if no text)
  complete <prompt>         Generate text completion for prompt
  batch <file.json|json>    Submit batch of requests from JSON
  help                      Show this help
        """
        console.print(Panel(help_text, border_style="blue"))

    def _run_chat_session(self):
        """Run interactive chat session with prompt_toolkit."""
        console.print("[cyan]Chat mode. Commands: /exit, /clear[/cyan]")

        messages = []
        chat_history_path = Path("~/.cache/levanter/chat_history").expanduser()
        chat_history_path.parent.mkdir(parents=True, exist_ok=True)
        chat_history = FileHistory(str(chat_history_path))
        chat_completer = WordCompleter(["/exit", "/clear"], ignore_case=True)

        while True:
            try:
                user_input = prompt("You: ", history=chat_history, completer=chat_completer).strip()

                if not user_input:
                    continue

                if user_input == "/exit":
                    break
                elif user_input == "/clear":
                    messages = []
                    console.print("[green]Conversation cleared[/green]")
                    continue

                messages.append(ChatMessage(role="user", content=user_input))
                response = self._run_chat_completion(messages, print_response=False)

                if response:
                    assistant_msg = response.choices[0].message.content
                    messages.append(ChatMessage(role="assistant", content=assistant_msg))
                    console.print(f"[green]Assistant:[/green] {assistant_msg}")

            except (KeyboardInterrupt, EOFError):
                break

    def batch(self, batch_input: str):
        """Submit a batch of chat completion requests from JSON.

        Args:
            batch_input: Either a JSON file path or inline JSON string
        """
        if not self.server:
            console.print("[red]No model loaded. Use 'load' command first[/red]")
            return

        # Try to load as file first
        if Path(batch_input).exists():
            with open(batch_input, "r") as f:
                batch_data = json.load(f)
            console.print(f"[blue]Loaded {len(batch_data)} requests from {batch_input}[/blue]")
        else:
            # Try to parse as inline JSON
            batch_data = json.loads(batch_input)
            console.print(f"[blue]Parsed {len(batch_data)} requests from inline JSON[/blue]")

        if not isinstance(batch_data, list):
            batch_data = [batch_data]

        # Submit all requests concurrently
        console.print(f"[cyan]Submitting batch of {len(batch_data)} requests...[/cyan]")

        start_time = time.time()

        server = self.server
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def submit_batch():
            tasks = []
            for i, req_data in enumerate(batch_data):
                # Convert to ChatCompletionRequest
                messages = []
                for msg in req_data.get("messages", []):
                    messages.append(ChatMessage(role=msg["role"], content=msg.get("content", "")))

                request = ChatCompletionRequest(
                    model=req_data.get("model", "<default model>"),
                    messages=messages,
                    max_tokens=req_data.get("max_tokens", self.config.max_tokens),
                    temperature=req_data.get("temperature", self.config.server.temperature),
                    n=req_data.get("n", 1),
                    logprobs=req_data.get("logprobs", False),
                    stop=req_data.get("stop"),
                )

                task = _create_chat_completion(server.inference_context, request)
                tasks.append(task)

            # Wait for all completions
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            return responses

        try:
            responses = loop.run_until_complete(submit_batch())
            elapsed = time.time() - start_time

            console.print(f"\n[green]Batch completed in {elapsed:.2f}s[/green]")
            for i, response in enumerate(responses):
                console.print(f"\n[bold]Response for Request {i + 1}:[/bold]")
                if isinstance(response, Exception):
                    console.print(f"[red]Error: {response}[/red]")
                else:
                    self._print_completion_response(response)
        finally:
            loop.close()

    def complete(self, prompt_text: str):
        """Generate a completion for a single string prompt."""
        if not self.server:
            console.print("[red]No model loaded. Use 'load' command first[/red]")
            return

        if not prompt_text:
            console.print("[red]Usage: complete <prompt>[/red]")
            return

        request = CompletionRequest(
            model=self.model_name or "model",
            prompt=prompt_text,
            max_tokens=self.config.max_tokens,
            temperature=self.config.server.temperature,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(_create_completion(self.server.inference_context, request))
            self._print_completion_response(response)
        finally:
            loop.close()

    def _run_chat_completion(self, messages, print_response=True):
        """Run async chat completion."""
        request = ChatCompletionRequest(
            model=self.model_name or "<default model>",
            messages=messages,
            stop=[self.server.inference_context.tokenizer.eos_token],
            max_tokens=self.config.max_tokens,
            temperature=self.config.server.temperature,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(_create_chat_completion(self.server.inference_context, request))

            if print_response:
                self._print_completion_response(response)
            return response
        finally:
            loop.close()

    def _print_completion_response(self, response):
        """Pretty print completion response."""
        for i, choice in enumerate(response.choices):
            if hasattr(choice, "message"):  # Chat completion
                content = choice.message.content
            else:  # Text completion
                content = choice.text

            console.print(f"Response {i + 1}/{len(response.choices)}: {content}")

        # Show usage stats
        if response.usage:
            console.print(
                f"[dim]Tokens - Prompt: {response.usage.prompt_tokens}, "
                f"Completion: {response.usage.completion_tokens}, "
                f"Total: {response.usage.total_tokens}[/dim]"
            )


def repl_mode(config: InferenceReplConfig, commands: ReplContext):
    """Run interactive REPL."""
    console.print(
        Panel.fit(
            "[bold blue]Levanter Inference REPL[/bold blue]\nType [bold]help[/bold] for commands",
            border_style="blue",
        )
    )

    # Auto-load model if specified
    if config.checkpoint:
        commands.load(config.checkpoint, config.tokenizer)

    # Setup prompt_toolkit
    command_names = list(commands.commands.keys()) + ["quit", "exit"]
    completer = WordCompleter(command_names, ignore_case=True)

    history_path = Path("~/.cache/levanter/inference_repl_history").expanduser()
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history = FileHistory(str(history_path))

    # REPL loop
    while True:
        try:
            user_input = prompt("🤖 > ", completer=completer, history=history).strip()

            if not user_input:
                continue

            # Parse command
            if user_input in ["quit", "exit"]:
                break

            parts = shlex.split(user_input)
            cmd = parts[0]
            args = parts[1:]

            # Handle commands
            if cmd == "load":
                # Parse: load path [--tokenizer tok]
                if not args:
                    console.print("[red]Usage: load <path> [--tokenizer <name>][/red]")
                    continue

                path = args[0]
                tokenizer = None
                if "--tokenizer" in args:
                    idx = args.index("--tokenizer")
                    tokenizer = args[idx + 1] if idx + 1 < len(args) else None
                commands.execute("load", path, tokenizer=tokenizer)
            elif cmd == "chat":
                message = " ".join(args) if args else None
                commands.execute("chat", message)
            elif cmd == "complete":
                if not args:
                    console.print("[red]Usage: complete <prompt>[/red]")
                    continue
                prompt_text = " ".join(args)
                commands.execute("complete", prompt_text)
            elif cmd in commands.commands:
                commands.execute(cmd)
            else:
                console.print(f"[red]Unknown command: {cmd}[/red]")
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    console.print("[blue]Goodbye![/blue]")


def cli_mode(config: InferenceReplConfig, commands: ReplContext):
    """Execute single command from CLI."""
    if not config.command:
        return

    if config.checkpoint:
        commands.load(config.checkpoint, config.tokenizer)
    else:
        console.print("[red]No model specified. Use --checkpoint to specify a model.[/red]")
        return

    if config.command == "chat":
        message = config.args if config.args else None
        commands.execute("chat", message)
    elif config.command == "complete":
        if not config.args:
            console.print("[red]Usage: complete <prompt>[/red]")
            return
        prompt_text = config.args
        commands.execute("complete", prompt_text)
    elif config.command == "batch":
        if not config.args:
            console.print("[red]Usage: batch <file.json> or batch <inline_json>[/red]")
            return
        commands.execute("batch", config.args)
    else:
        commands.execute(config.command, *config.args)


def main(config: InferenceReplConfig):
    """Main entry point."""
    commands = ReplContext(config)
    os.environ["EQX_ON_ERROR"] = "nan"

    # Determine mode
    if config.command:
        cli_mode(config, commands)
    else:
        repl_mode(config, commands)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    levanter.config.main(main)()
