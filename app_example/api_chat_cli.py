import argparse
import sys
import requests


def ask(url: str, prompt: str, timeout: int = 300) -> str:
    r = requests.post(url, json={"prompt": prompt}, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("result", str(data))


def main():
    ap = argparse.ArgumentParser(description="Qwen API simple CLI client")
    ap.add_argument("--url", default="http://127.0.0.1:8000/generate", help="API endpoint URL")
    ap.add_argument("--once", default=None, help="Ask once and exit")
    args = ap.parse_args()

    if args.once:
        try:
            print(ask(args.url, args.once))
        except Exception as e:
            print(f"[Error] {e}", file=sys.stderr)
            sys.exit(1)
        return

    print(f"API Chat ({args.url})  空行で終了")
    while True:
        try:
            q = input("> ").strip()
            if not q:
                break
            print(ask(args.url, q))
        except KeyboardInterrupt:
            print("\nbye.")
            break
        except Exception as e:
            print(f"[Error] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

