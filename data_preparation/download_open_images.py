import os
import subprocess
import sys

def main():
    # 1. Define and create the output directory securely
    output_dir = os.path.join("datasets", "open_images")
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Verified output directory: {output_dir}")
    except Exception as e:
        print(f"❌ Failed to create directory {output_dir}: {e}")
        sys.exit(1)

    print("\n🚀 Starting Open Images Dataset Download...")
    print("Class: Person | Limit: 4000 images")
    
    import shutil
    
    # 2. Find the oidv6 executable explicitly in case it's not in PATH
    oidv6_cmd = "oidv6"
    if not shutil.which("oidv6"):
        # Attempt to locate standard Windows --user install path
        user_script_path = os.path.join(os.environ.get("APPDATA", ""), "Python", "Python311", "Scripts", "oidv6.exe")
        if os.path.exists(user_script_path):
            oidv6_cmd = f'"{user_script_path}"'
        else:
            print("❌ Error: oidv6 not found in PATH or standard user directories. Please add it to your PATH.")
            sys.exit(1)

    command = (
        f'{oidv6_cmd} downloader en --dataset "{output_dir}" '
        f'--type_data train --classes Person --limit 4000 --multi_classes 1 --yes'
    )
    
    print(f"⚙️  Executing: {command}\n")
    print("-" * 50)
    
    # 3. Execute the download command and handle missing dependencies safely
    try:
        # shell=True is highly recommended on Windows for CLI wrappers like oidv6
        result = subprocess.run(command, shell=True, check=True)
        print("-" * 50)
        print("\n✅ Dataset download completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print("-" * 50)
        print(f"\n❌ Error: The oidv6 downloader failed with exit code {e.returncode}.")
        print("Please ensure you have enough disk space and a stable internet connection.")
        sys.exit(e.returncode)
        
    except FileNotFoundError:
        print("-" * 50)
        print("\n❌ Error: The 'oidv6' command was not found.")
        print("You must install the package first. Run:")
        print("    pip install oidv6")
        sys.exit(1)
        
    except Exception as e:
        print("-" * 50)
        print(f"\n❌ An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
