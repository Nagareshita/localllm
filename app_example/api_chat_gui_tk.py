import os
import threading
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import requests


URL = os.environ.get("QWEN_API_URL", "http://127.0.0.1:8000/generate")


class App:
    def __init__(self, root: tk.Tk):
        root.title("Qwen API Chat")
        self.root = root

        self.txt = ScrolledText(root, width=80, height=24, state="disabled")
        self.txt.pack(padx=8, pady=8, fill="both", expand=True)

        frame = tk.Frame(root)
        frame.pack(padx=8, pady=(0, 8), fill="x")

        self.entry = tk.Entry(frame)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda e: self.send())

        self.btn = tk.Button(frame, text="送信", command=self.send)
        self.btn.pack(side="left", padx=(8, 0))

        self.append("INFO", f"Endpoint: {URL}")

    def append(self, who: str, text: str):
        self.txt.configure(state="normal")
        self.txt.insert("end", f"{who}: {text}\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def send(self):
        prompt = self.entry.get().strip()
        if not prompt:
            return
        self.entry.delete(0, "end")
        self.append("You", prompt)
        self.btn.config(state="disabled")
        threading.Thread(target=self._call_api, args=(prompt,), daemon=True).start()

    def _call_api(self, prompt: str):
        try:
            r = requests.post(URL, json={"prompt": prompt}, timeout=300)
            r.raise_for_status()
            data = r.json()
            out = data.get("result", str(data))
        except Exception as e:
            out = f"[Error] {e}"
        self.root.after(0, lambda: (self.append("AI", out), self.btn.config(state="normal")))


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()

