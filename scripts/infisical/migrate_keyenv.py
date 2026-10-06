"""Copy manifest-bound native Keychain credentials to linked Infisical development."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import ROOT, Infisical, SecretError, cli_environment, matches


def keyenv_python():
    executable = shutil.which("keyenv")
    if not executable:
        raise SecretError("Install keyenv before migrating its Keychain accounts.")
    shebang = Path(executable).read_text().splitlines()[0]
    interpreter = shebang[2:]
    has_interpreter = shebang.startswith("#!/") and Path(interpreter).is_file()
    if not has_interpreter:
        raise SecretError("Could not locate keyenv's installed Python interpreter.")
    return interpreter


def read_keychain(root=ROOT):
    max_payload_size = 1_048_576
    read_fd, write_fd = os.pipe()
    process = None
    try:
        process = subprocess.Popen(
            [
                keyenv_python(),
                str(Path(__file__).with_name("keychain_reader.py")),
                "--manifest",
                str(root / ".keyenv.toml"),
                "--output-fd",
                str(write_fd),
            ],
            pass_fds=(write_fd,),
            stdout=subprocess.DEVNULL,
            env=cli_environment(),
        )
    except Exception:
        os.close(read_fd)
        raise SecretError("Could not start the native Keychain reader.") from None
    finally:
        os.close(write_fd)
    with os.fdopen(read_fd, "r") as pipe:
        payload = pipe.read(max_payload_size + 1)

    if len(payload) > max_payload_size:
        process.kill()
        process.wait()
        raise SecretError("Keychain response exceeded the permitted size.")
    if process.wait():
        raise SecretError("Keychain read failed; originals retained.")
    try:
        return json.loads(payload)
    except Exception:
        raise SecretError("Invalid Keychain response; raw output suppressed.") from None


def transfer(source, destination):
    source_keys = source["keys"]
    blocked_keys = [
        key
        for key, item in source_keys.items()
        if item["status"] not in ("available", "missing")
    ]
    if blocked_keys:
        raise SecretError(
            "Keychain bindings are missing or unauthorized; resolve them before migration."
        )
    values = {
        key: item["value"]
        for key, item in source_keys.items()
        if item["status"] == "available"
    }
    # Check every available key before creating any. Never replace differing values.
    existing = destination.read()
    for key, value in values.items():
        if key in existing and not matches(existing[key], value):
            raise SecretError(
                f"Different existing value for {key}; no credentials were written."
            )
    report = {}
    for key, item in source_keys.items():
        if key in values:
            report[key] = destination.create_and_verify(key, item["value"])
        else:
            report[key] = "absent from keyenv; skipped"
        print(f"{key}: {report[key]}", flush=True)
    return report


def main():
    destination = Infisical()
    destination.read()  # Authenticate before asking Keychain for any values.
    transfer(read_keychain(), destination)
    print(
        "Transfer verified. Keychain originals retained. No secret values were displayed."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Migration cancelled; Keychain originals retained.", file=sys.stderr)
        sys.exit(130)
    except SecretError as error:
        print(error, file=sys.stderr)
        sys.exit(1)
    except Exception:
        print(
            "Migration failed; credential details suppressed and Keychain originals retained.",
            file=sys.stderr,
        )
        sys.exit(1)
