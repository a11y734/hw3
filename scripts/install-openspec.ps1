param(
    [string]$Version = "latest"
)

Write-Host "Installing openspec ($Version) ..."
if ($Version -eq "latest") {
    pip install openspec
} else {
    pip install "openspec==$Version"
}

Write-Host "openspec version:"
openspec --version
