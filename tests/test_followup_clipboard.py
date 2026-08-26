import unittest
from unittest.mock import patch

import pyperclip

import app


class FollowupClipboardTests(unittest.TestCase):
    @patch("app._append_clipboard_to_followup")
    def test_does_not_read_clipboard_when_user_leaves_toggle_off(self, append):
        history = [{"role": "user", "content": "继续解释"}]

        result = app._prepare_followup_history(history, include_clipboard=False)

        self.assertEqual(result, history)
        append.assert_not_called()

    @patch("app._append_clipboard_to_followup")
    def test_reads_clipboard_when_user_turns_toggle_on(self, append):
        history = [{"role": "user", "content": "继续解释"}]
        append.return_value = history

        result = app._prepare_followup_history(history, include_clipboard=True)

        self.assertEqual(result, history)
        append.assert_called_once_with(history)

    @patch("app.pyperclip.paste", return_value="print('from desktop')")
    def test_appends_desktop_clipboard_to_latest_followup(self, _paste):
        history = [
            {"role": "assistant", "content": "原解答"},
            {"role": "user", "content": "解释这段代码"},
        ]

        result = app._append_clipboard_to_followup(history)

        self.assertEqual(result[0]["content"], "原解答")
        self.assertEqual(
            result[-1]["content"],
            "解释这段代码\n\n"
            "[电脑当前文本剪贴板内容]\n"
            "print('from desktop')",
        )

    @patch("app.pyperclip.paste", return_value="   ")
    def test_ignores_empty_clipboard(self, _paste):
        history = [{"role": "user", "content": "继续解释"}]

        result = app._append_clipboard_to_followup(history)

        self.assertEqual(result, history)
        self.assertEqual(result[-1]["content"], "继续解释")

    @patch(
        "app.pyperclip.paste",
        side_effect=pyperclip.PyperclipException("clipboard unavailable"),
    )
    def test_clipboard_failure_does_not_block_followup(self, _paste):
        history = [{"role": "user", "content": "继续解释"}]

        result = app._append_clipboard_to_followup(history)

        self.assertEqual(result, history)
        self.assertEqual(result[-1]["content"], "继续解释")

    @patch("app.pyperclip.paste")
    def test_truncates_large_clipboard(self, paste):
        paste.return_value = "x" * (app.MAX_CLIPBOARD_CHARS + 1)
        history = [{"role": "user", "content": "继续解释"}]

        result = app._append_clipboard_to_followup(history)
        clipboard = result[-1]["content"].split(
            "[电脑当前文本剪贴板内容]\n", 1
        )[1]

        self.assertEqual(len(clipboard), app.MAX_CLIPBOARD_CHARS)


if __name__ == "__main__":
    unittest.main()
