def has_uncomitted_changes():
    return subprocess.check_output(["git", "status", "-s"])
