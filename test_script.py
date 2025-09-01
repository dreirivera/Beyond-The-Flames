import subprocess

def test_tool(name, command):
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        print(f"\n{name} output:\n{result.stdout}")
        if result.returncode == 0:
            print(f"✅ {name} is callable and working.")
        else:
            print(f"⚠️ {name} ran with exit code {result.returncode} (still callable).")
    except FileNotFoundError:
        print(f"❌ {name} not found. Check if it's in your PATH.")

test_tool("FDS", ["fds", "test.fds"])  
test_tool("Smokeview", ["smokeview", "-version"])
