# Define the URL and destination paths
$url = "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip"
$destinationZip = "data/energy+efficiency.zip"
$extractToFolder = "data"

# Check if the destination folder exists, if not, create it
if (!(Test-Path -Path $extractToFolder)) {
    New-Item -ItemType Directory -Force -Path $extractToFolder
}

# Download the file
Invoke-WebRequest -Uri $url -OutFile $destinationZip

# Extract the ZIP file
Expand-Archive -Path $destinationZip -DestinationPath $extractToFolder

# Remove the ZIP file
Remove-Item $destinationZip

# Output message
Write-Host "Download and extraction complete."
