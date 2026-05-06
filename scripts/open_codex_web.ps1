param(
    [string]$Prompt = "",
    [string]$ProjectDir = ""
)

$ErrorActionPreference = "Stop"

function Resolve-CodexExe {
    $cmd = Get-Command codex -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -and (Test-Path $cmd.Source)) {
        return $cmd.Source
    }

    $extensionRoots = @(
        (Join-Path $env:USERPROFILE ".vscode\extensions"),
        (Join-Path $env:USERPROFILE ".cursor\extensions")
    )

    foreach ($root in $extensionRoots) {
        if (-not (Test-Path $root)) {
            continue
        }

        $candidate = Get-ChildItem $root -Directory -Filter "openai.chatgpt-*-win32-x64" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            ForEach-Object { Join-Path $_.FullName "bin\windows-x86_64\codex.exe" } |
            Where-Object { Test-Path $_ } |
            Select-Object -First 1

        if ($candidate) {
            return $candidate
        }
    }

    throw "codex.exe nao encontrado."
}

if ([string]::IsNullOrWhiteSpace($ProjectDir)) {
    $ProjectDir = (Get-Location).Path
}

$codexExe = Resolve-CodexExe

Write-Host "Usando Codex em: $codexExe"
Write-Host "Projeto: $ProjectDir"

if ([string]::IsNullOrWhiteSpace($Prompt)) {
    & $codexExe --search -C $ProjectDir
} else {
    & $codexExe --search -C $ProjectDir $Prompt
}
