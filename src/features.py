"""Human-readable names for the 2381 EMBER (feature-version 2) columns.

EMBER ships PE files as a fixed vector. Several blocks are *hashed* (imports, section names, header enums), so a
column index alone says nothing to an analyst. Here we:
  * name every column by the block it belongs to,
  * map each block to one of the project's feature families (PE header / opcode proxy / API calls / strings),
  * invert the feature hash for well-known Windows APIs, section names and header enums, so that e.g. column
    "imp_func_417" reads "~kernel32.dll:VirtualAllocEx". Hash collisions are possible, so such names are *hints*.
"""
from functools import lru_cache

from sklearn.feature_extraction import FeatureHasher

# (block, width) in the exact order of ember.features.PEFeatureExtractor (v2)
BLOCKS = [("byte_hist", 256), ("byte_entropy", 256), ("strings", 104), ("general", 10), ("header", 62),
          ("section", 255), ("imports", 1280), ("exports", 128), ("datadir", 30)]
N_FEATURES = sum(w for _, w in BLOCKS)  # 2381

FAMILY = {"byte_hist": "Opcode proxy (byte freq.)", "byte_entropy": "Opcode proxy (byte freq.)",
          "imports": "API calls", "exports": "API calls", "strings": "Strings",
          "general": "PE header", "header": "PE header", "section": "PE header", "datadir": "PE header"}

# Single-byte x86 opcodes that dominate compiled code; used to label byte-histogram bins.
OPCODES = {0x00: "ADD/zero-padding", 0x0F: "two-byte escape (Jcc/MOVZX/SETcc)", 0x33: "XOR r,r/m", 0x39: "CMP r/m,r",
           0x3B: "CMP r,r/m", 0x50: "PUSH eax", 0x51: "PUSH ecx", 0x52: "PUSH edx", 0x53: "PUSH ebx",
           0x55: "PUSH ebp", 0x56: "PUSH esi", 0x57: "PUSH edi", 0x58: "POP eax", 0x59: "POP ecx", 0x5D: "POP ebp",
           0x5E: "POP esi", 0x5F: "POP edi", 0x68: "PUSH imm32", 0x6A: "PUSH imm8", 0x74: "JZ", 0x75: "JNZ",
           0x83: "ALU r/m,imm8", 0x85: "TEST r/m,r", 0x89: "MOV r/m,r", 0x8B: "MOV r,r/m", 0x8D: "LEA",
           0x90: "NOP", 0xA1: "MOV eax,[addr]", 0xB8: "MOV eax,imm32", 0xC2: "RET imm16", 0xC3: "RET",
           0xC7: "MOV r/m,imm32", 0xC9: "LEAVE", 0xCC: "INT3/padding", 0xCD: "INT", 0xE8: "CALL rel32",
           0xE9: "JMP rel32", 0xEB: "JMP rel8", 0xFF: "CALL/JMP/PUSH r/m (group 5)", 0xFE: "INC/DEC r/m8"}

# Windows APIs whose import is informative for malware triage (library -> functions).
APIS = {
    "kernel32.dll": """VirtualAlloc VirtualAllocEx VirtualProtect VirtualProtectEx WriteProcessMemory ReadProcessMemory
        CreateRemoteThread CreateThread OpenProcess CreateProcessA CreateProcessW WinExec GetProcAddress LoadLibraryA
        LoadLibraryW GetModuleHandleA GetModuleHandleW IsDebuggerPresent CheckRemoteDebuggerPresent
        Sleep GetTickCount CreateFileA CreateFileW WriteFile ReadFile DeleteFileA DeleteFileW CopyFileA MoveFileA
        FindFirstFileA FindNextFileA GetTempPathA GetSystemDirectoryA GetWindowsDirectoryA CreateToolhelp32Snapshot
        Process32First Process32Next Module32First TerminateProcess ExitProcess GetCommandLineA GetStartupInfoA
        GetLastError CloseHandle HeapAlloc HeapFree GetModuleFileNameA LockResource LoadResource FindResourceA
        SizeofResource GetComputerNameA GetVersionExA CreateMutexA OpenMutexA SetFileAttributesA GetCurrentProcess
        SetUnhandledExceptionFilter WideCharToMultiByte MultiByteToWideChar lstrcpyA lstrlenA""".split(),
    "advapi32.dll": """RegOpenKeyExA RegOpenKeyExW RegSetValueExA RegSetValueExW RegCreateKeyExA RegCreateKeyExW
        RegQueryValueExA RegDeleteKeyA RegCloseKey OpenProcessToken AdjustTokenPrivileges LookupPrivilegeValueA
        CreateServiceA StartServiceA OpenSCManagerA CryptAcquireContextA CryptEncrypt CryptDecrypt CryptGenKey
        GetUserNameA""".split(),
    "user32.dll": """SetWindowsHookExA SetWindowsHookExW GetAsyncKeyState GetKeyState GetForegroundWindow FindWindowA
        MessageBoxA ShowWindow GetDC SendMessageA PostMessageA RegisterClassA CreateWindowExA SystemParametersInfoA
        ExitWindowsEx GetMessageA DispatchMessageA""".split(),
    "wininet.dll": """InternetOpenA InternetOpenUrlA InternetReadFile InternetConnectA HttpOpenRequestA HttpSendRequestA
        InternetOpenW FtpPutFileA""".split(),
    "ws2_32.dll": "WSAStartup socket connect send recv bind listen accept gethostbyname closesocket".split(),
    "urlmon.dll": "URLDownloadToFileA URLDownloadToFileW".split(),
    "shell32.dll": "ShellExecuteA ShellExecuteW SHGetFolderPathA SHGetSpecialFolderPathA".split(),
    "ntdll.dll": "NtQueryInformationProcess NtUnmapViewOfSection RtlDecompressBuffer ZwQuerySystemInformation".split(),
    "msvcrt.dll": "malloc free memcpy strcpy strlen printf exit _exit".split(),
    "ole32.dll": "CoInitialize CoCreateInstance".split(),
    "gdi32.dll": "BitBlt CreateCompatibleDC GetPixel".split(),
    "comctl32.dll": "InitCommonControls".split(),
    "mscoree.dll": "_CorExeMain _CorDllMain".split(),
    "version.dll": "GetFileVersionInfoA VerQueryValueA".split(),
}
LIBS = list(APIS) + ["psapi.dll", "netapi32.dll", "crypt32.dll", "oleaut32.dll", "winmm.dll", "shlwapi.dll",
                     "mpr.dll", "iphlpapi.dll", "imm32.dll", "comdlg32.dll", "wsock32.dll", "msvbvm60.dll"]
SECTIONS = [".text", ".data", ".rdata", ".rsrc", ".reloc", ".idata", ".edata", ".bss", ".tls", ".pdata", "UPX0",
            "UPX1", "UPX2", ".aspack", ".adata", ".petite", ".themida", ".vmp0", ".vmp1", ".nsp0", ".nsp1", "CODE",
            "DATA", "BSS", ".code", ".crt", ".ndata", ".MPRESS1", ".MPRESS2", "text", ".sdata", ".textbss"]
MACHINES = ["I386", "AMD64", "IA64", "ARM", "ARMNT", "ARM64", "UNKNOWN"]
SUBSYSTEMS = ["WINDOWS_GUI", "WINDOWS_CUI", "NATIVE", "EFI_APPLICATION", "UNKNOWN", "POSIX_CUI"]
MAGICS = ["PE32", "PE32_PLUS"]


@lru_cache(maxsize=None)
def _hasher(n):
    return FeatureHasher(n, input_type="string")


def hash_bin(n, token):
    """Column (0..n-1) that EMBER's FeatureHasher assigns to `token`."""
    return int(_hasher(n).transform([[token]]).nonzero()[1][0])


def _reverse(n, tokens, fmt=str):
    out = {}
    for t in tokens:
        out.setdefault(hash_bin(n, t), []).append(fmt(t))
    return {k: "|".join(v[:3]) + ("|…" if len(v) > 3 else "") for k, v in out.items()}


@lru_cache(maxsize=None)
def feature_names():
    names = []
    names += [f"byte_0x{b:02X}" + (f" ({OPCODES[b]})" if b in OPCODES else "") for b in range(256)]
    names += [f"entropy_bin{h:02d}_hinibble{x:X}" for h in range(16) for x in range(16)]
    names += ["str_numstrings", "str_avg_length", "str_printables"]
    names += [f"str_char_0x{c:02X}('{chr(c)}')" for c in range(32, 128)]
    names += ["str_entropy", "str_paths(C:\\)", "str_urls(http://)", "str_registry(HKEY_)", "str_MZ_headers"]
    names += ["file_size", "virtual_size", "has_debug", "num_exports", "num_imports", "has_relocations",
              "has_resources", "has_signature", "has_tls", "num_symbols"]
    mach = _reverse(10, MACHINES)
    sub = _reverse(10, SUBSYSTEMS)
    mag = _reverse(10, MAGICS)
    names += ["hdr_timestamp"]
    names += [f"hdr_machine_{i}" + (f" (~{mach[i]})" if i in mach else "") for i in range(10)]
    names += [f"hdr_coff_characteristics_{i}" for i in range(10)]
    names += [f"hdr_subsystem_{i}" + (f" (~{sub[i]})" if i in sub else "") for i in range(10)]
    names += [f"hdr_dll_characteristics_{i}" for i in range(10)]
    names += [f"hdr_magic_{i}" + (f" (~{mag[i]})" if i in mag else "") for i in range(10)]
    names += ["hdr_major_image_ver", "hdr_minor_image_ver", "hdr_major_linker_ver", "hdr_minor_linker_ver",
              "hdr_major_os_ver", "hdr_minor_os_ver", "hdr_major_subsystem_ver", "hdr_minor_subsystem_ver",
              "hdr_sizeof_code", "hdr_sizeof_headers", "hdr_sizeof_heap_commit"]
    entry = _reverse(50, SECTIONS)
    names += ["sec_count", "sec_nonzero_size", "sec_empty_name", "sec_read_exec", "sec_writable"]
    names += [f"sec_size_{i}" for i in range(50)] + [f"sec_entropy_{i}" for i in range(50)]
    names += [f"sec_vsize_{i}" for i in range(50)]
    names += [f"sec_entry_{i}" + (f" (~{entry[i]})" if i in entry else "") for i in range(50)]
    names += [f"sec_characteristics_{i}" for i in range(50)]
    libs = _reverse(256, LIBS)
    funcs_full = _reverse(1024, [f"{lib}:{fn}" for lib, fns in APIS.items() for fn in fns])
    names += [f"imp_lib_{i}" + (f" (~{libs[i]})" if i in libs else "") for i in range(256)]
    names += [f"imp_api_{i}" + (f" (~{funcs_full[i]})" if i in funcs_full else "") for i in range(1024)]
    names += [f"exp_{i}" for i in range(128)]
    dd = ["EXPORT", "IMPORT", "RESOURCE", "EXCEPTION", "CERTIFICATE", "BASERELOC", "DEBUG", "ARCH", "GLOBALPTR",
          "TLS", "LOAD_CONFIG", "BOUND_IMPORT", "IAT", "DELAY_IMPORT", "CLR"]
    names += [f"dd_{d}_{k}" for d in dd for k in ("size", "vaddr")]
    assert len(names) == N_FEATURES, len(names)
    return names


@lru_cache(maxsize=None)
def feature_families():
    fam = []
    for block, width in BLOCKS:
        fam += [FAMILY[block]] * width
    return fam


def family_slices():
    """family -> sorted column indices"""
    out = {}
    for i, f in enumerate(feature_families()):
        out.setdefault(f, []).append(i)
    return out


if __name__ == "__main__":
    n = feature_names()
    print(len(n), "features;", sum("~" in x for x in n), "with decoded hints")
    print([x for x in n if x.startswith("imp_api") and "~" in x][:8])
