"""Command-line interface for the bert-chatbot package."""

import typer

from bert_chatbot.commands import chat_cmd

app = typer.Typer(
    help="BERT Semantic ChatBot.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["--help", "-h"]},
)

app.add_typer(chat_cmd.app, name="chat", help="Interactive chat with the semantic chatbot.")

if __name__ == "__main__":
    app() 