import json, random
from src import config

OS_VERSIONS = ["Windows 10", "Windows 11", "Windows Server 2019", "Windows Server 2022"]

TEMPLATES = [
    ("Windows Update", "update installation fails partway through",
     ["corrupted update cache", "insufficient disk space on the system partition",
      "a blocked connection to the update servers", "a pending reboot from a prior update"],
     ["Clear the SoftwareDistribution folder", "Run the Update troubleshooter",
      "Restart the Windows Update service", "Run DISM RestoreHealth followed by SFC scannow"]),
    ("Windows Boot", "the system fails to reach the desktop and shows a stop screen",
     ["a corrupted boot configuration entry", "a driver loaded too early in the boot sequence",
      "a damaged system file", "an incomplete update applied at shutdown"],
     ["Boot into recovery and run Startup Repair", "Rebuild the boot configuration data",
      "Uninstall the most recent update from recovery", "Boot in safe mode and roll back the driver"]),
    ("Device Drivers", "a device stops responding after a driver update",
     ["an unsigned driver package", "a version mismatch with the current OS build",
      "a conflicting driver left from a prior install", "a failed digital signature check"],
     ["Roll back the driver in Device Manager", "Remove the old driver package with pnputil",
      "Install the vendor-signed driver manually", "Disable driver signature enforcement temporarily"]),
    ("Networking", "network connectivity drops intermittently",
     ["a stale DNS cache", "a misconfigured proxy setting",
      "a corrupted TCP/IP stack", "an adapter power-saving setting"],
     ["Flush the DNS cache", "Reset the TCP/IP stack with netsh",
      "Disable power management on the adapter", "Clear the proxy configuration"]),
    ("Disk and Filesystem", "file operations fail with an I/O error",
     ["filesystem corruption on the volume", "bad sectors on the physical disk",
      "an interrupted write during shutdown", "a failing storage controller driver"],
     ["Run chkdsk with repair flags", "Check disk health with SMART diagnostics",
      "Restore the affected files from backup", "Update the storage controller driver"]),
    ("Office Applications", "the application closes unexpectedly when opening a document",
     ["a faulty add-in", "a corrupted user profile",
      "an incomplete application update", "a damaged template file"],
     ["Start the application in safe mode", "Disable all add-ins and re-enable one at a time",
      "Run an online repair of the installation", "Rename the default template to force a rebuild"]),
    ("Activation and Licensing", "the system reports it is not activated",
     ["a hardware change invalidating the license binding", "a licensing service that failed to start",
      "an unreachable activation server", "a mismatched product key edition"],
     ["Run the activation troubleshooter", "Restart the licensing service",
      "Re-enter the product key with slmgr", "Verify the edition matches the key"]),
    ("Printing", "print jobs remain queued and never complete",
     ["a stalled print spooler service", "a corrupted print job in the queue",
      "an outdated printer driver", "a lost connection to the print server"],
     ["Stop the spooler and clear the print queue folder", "Restart the print spooler service",
      "Reinstall the printer driver", "Re-add the printer connection"]),
    (".NET Framework", "an application fails to launch with a runtime error",
     ["a missing framework component", "a corrupted framework installation",
      "a version mismatch with the application target", "a failed security update"],
     ["Repair the framework installation", "Install the required framework version",
      "Run the framework repair tool", "Re-register the framework with the runtime"]),
    ("Windows Store", "app downloads fail to complete",
     ["a corrupted store cache", "an out-of-sync licensing state",
      "insufficient space in the app install location", "a signed-out account state"],
     ["Reset the store cache with wsreset", "Sign out and back into the store account",
      "Re-register the store app package", "Free space on the install drive"]),
    ("Permissions", "access to a folder is denied despite administrator rights",
     ["broken ownership after a profile migration", "inherited permissions removed by policy",
      "a locked file held by a background service", "a corrupted access control list"],
     ["Take ownership of the folder", "Reset permissions inheritance",
      "Identify and stop the locking process", "Restore the default access control list"]),
    ("Memory and Performance", "the system becomes unresponsive under moderate load",
     ["a memory leak in a background service", "insufficient page file size",
      "a faulty memory module", "excessive startup applications"],
     ["Identify the leaking process in Resource Monitor", "Increase the page file size",
      "Run Windows Memory Diagnostic", "Disable unnecessary startup entries"]),
]


def make_code(rng):
    return "0x8" + "".join(rng.choice("0123456789ABCDEF") for _ in range(7))


def generate(n=config.N_ERRORS, seed=config.SEED):
    rng = random.Random(seed)
    errors, seen = [], set()
    while len(errors) < n:
        product, symptom, causes, fixes = rng.choice(TEMPLATES)
        code = make_code(rng)
        if code in seen:
            continue
        seen.add(code)
        steps = rng.sample(fixes, 3)
        errors.append({
            "error_code": code,
            "os_version": rng.choice(OS_VERSIONS),
            "product": product,
            "symptom": f"Error {code}: {symptom}.",
            "cause": f"This typically occurs due to {rng.choice(causes)}.",
            "solution": " ".join(f"{i}. {s}." for i, s in enumerate(steps, 1)),
        })
    return errors


if __name__ == "__main__":
    errors = generate()
    config.DATA_FILE.parent.mkdir(exist_ok=True)
    with open(config.DATA_FILE, "w") as f:
        json.dump(errors, f, indent=2)
    print(f"wrote {len(errors)} errors to {config.DATA_FILE}")