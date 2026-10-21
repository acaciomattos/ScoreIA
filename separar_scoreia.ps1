<#
Uso: .\separar_scoreia.ps1 -Audio "C:\caminho\Faixa Vermelha suno.wav"
Reutiliza o Demucs do ScoreIA; nao instala nem atualiza dependencias.
Na primeira utilizacao, o Demucs pode baixar os pesos do modelo.
#>
[CmdletBinding()]
param(
    [string]$Audio = (Join-Path $PSScriptRoot 'Faixa Vermelha suno.wav'),
    [string]$ScoreRoot = 'C:\Users\acaci\Documents\scoreIA\ScoreIA',
    [string]$Python,
    [string]$Saida
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Audio -PathType Leaf)) {
    throw "WAV nao encontrado: $Audio. Baixe o anexo e informe seu caminho com -Audio."
}
$Audio = (Resolve-Path -LiteralPath $Audio).Path
if ([IO.Path]::GetExtension($Audio) -ine '.wav') { throw 'Informe o arquivo WAV.' }
if (-not $Python) {
    foreach ($ambiente in @('.venv310', '.venv312', '.venv')) {
        $candidato = Join-Path $ScoreRoot "$ambiente\Scripts\python.exe"
        $modulo = Join-Path $ScoreRoot "$ambiente\Lib\site-packages\demucs\separate.py"
        if ((Test-Path -LiteralPath $candidato) -and (Test-Path -LiteralPath $modulo)) {
            $Python = $candidato
            break
        }
    }
}
if (-not $Python -or -not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Python com Demucs nao encontrado no ScoreIA. Informe -Python com o caminho do python.exe do ambiente correto.'
}
if (-not $Saida) { $Saida = Join-Path $PSScriptRoot 'separated' }
# Cada execucao ganha uma pasta para preservar resultados anteriores.
$execucao = Join-Path $Saida ([guid]::NewGuid().ToString('N'))
$null = New-Item -ItemType Directory -Path $execucao
$execucao = (Resolve-Path -LiteralPath $execucao).Path
$argumentos = @('-m', 'demucs.separate', '-n', 'mdx_extra_q',
    '--two-stems', 'vocals', '-d', 'cpu', '--segment', '10',
    '--int24', '-j', '1', '-o', $execucao, $Audio)
Write-Host "Python: $Python"
Write-Host 'Separando vocal e instrumental com mdx_extra_q em CPU...'
& $Python @argumentos
if ($LASTEXITCODE -ne 0) { throw "Demucs falhou (codigo $LASTEXITCODE). Veja a mensagem acima." }
$pasta = Join-Path (Join-Path $execucao 'mdx_extra_q') ([IO.Path]::GetFileNameWithoutExtension($Audio))
$vocal = Join-Path $pasta 'vocals.wav'
$instrumental = Join-Path $pasta 'no_vocals.wav'
foreach ($arquivo in @($vocal, $instrumental)) {
    if (-not (Test-Path -LiteralPath $arquivo -PathType Leaf)) { throw "Saida ausente: $arquivo" }
    if ((Get-Item -LiteralPath $arquivo).Length -le 44) { throw "Saida vazia: $arquivo" }
}
Rename-Item -LiteralPath $instrumental -NewName 'instrumental.wav'
Write-Host "Vocal:        $vocal"
Write-Host "Instrumental: $(Join-Path $pasta 'instrumental.wav')"
