# Targeted Download for QuantumBioSim Ancestral Verification
# Dataset: Kap København Formation (2 Million Year Old eDNA) & Neanderthal Genome

$destDir = "data/ancestral_fastq"
New-Item -ItemType Directory -Force -Path $destDir | Out-Null

function Save-AncestralSample {
    param (
        [string]$Url,
        [string]$OutputPath,
        [string]$Description,
        [int]$ReadIndex = 0 # 0 for _1, 1 for _2
    )

    $fileName = Split-Path $OutputPath -Leaf
    Write-Host "Checking $Description..."

    # 1. Check if file exists
    if (Test-Path $OutputPath) {
        Write-Host "   File exists. Verifying integrity..."
        
        # 2. Fetch MD5 from ENA API (Reliable)
        try {
            $runAcc = $fileName.Split("_")[0]
            $apiUrl = "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=$runAcc&result=read_run&fields=fastq_md5&format=json"
            
            $json = Invoke-RestMethod -Uri $apiUrl -ErrorAction Stop
            # API returns array of objects. Get first item.
            # fastq_md5 field format: "md5_file1;md5_file2" (for paired reads)
            $remoteHashes = $json[0].fastq_md5.Split(";")
            
            if ($ReadIndex -lt $remoteHashes.Count) {
                $enaHash = $remoteHashes[$ReadIndex].Trim()
                 
                # Calculate Local Hash
                $localHashObj = Get-FileHash -Path $OutputPath -Algorithm MD5
                $localHash = $localHashObj.Hash.ToLower()
                
                if ($localHash -eq $enaHash) {
                    Write-Host "   [MATCH] File is complete and verified. Skipping download."
                    return
                }
                else {
                    Write-Host "   [MISMATCH] Local MD5: $localHash != Remote: $enaHash"
                    Write-Host "   Re-downloading..."
                }
            }
            else {
                Write-Host "   [WARNING] API did not return hash for ReadIndex $ReadIndex. Skipping verification."
            }
        }
        catch {
            Write-Host "   [WARNING] API Hash Check Failed ($_.Exception.Message). Defaulting to re-download to ensure safety."
        }
    }

    Write-Host "   Downloading $Description..."
    # Use Start-BitsTransfer for robust handling of large files (prevents timeouts)
    Start-BitsTransfer -Source $Url -Destination $OutputPath
    Write-Host "   Download Complete."
}

Write-Host "Starting Targeted Download (Paired-End Complete)..."

# 1. Ancient Fungi (Prototaxites Link) - Run ERR10493281
Save-AncestralSample -Url "https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR104/081/ERR10493281/ERR10493281_1.fastq.gz" `
    -OutputPath "$destDir/ERR10493281_1.fastq.gz" `
    -Description "Run ERR10493281 (Ancient Fungi) - Read 1" `
    -ReadIndex 0

Save-AncestralSample -Url "https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR104/081/ERR10493281/ERR10493281_2.fastq.gz" `
    -OutputPath "$destDir/ERR10493281_2.fastq.gz" `
    -Description "Run ERR10493281 (Ancient Fungi) - Read 2" `
    -ReadIndex 1

# 2. Ancestral Eukaryote (Human Lineage Link) - Run ERR10493300
Save-AncestralSample -Url "https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR104/000/ERR10493300/ERR10493300_1.fastq.gz" `
    -OutputPath "$destDir/ERR10493300_1.fastq.gz" `
    -Description "Run ERR10493300 (Ancestral Eukaryote) - Read 1" `
    -ReadIndex 0

Save-AncestralSample -Url "https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR104/000/ERR10493300/ERR10493300_2.fastq.gz" `
    -OutputPath "$destDir/ERR10493300_2.fastq.gz" `
    -Description "Run ERR10493300 (Ancestral Eukaryote) - Read 2" `
    -ReadIndex 1

# 3. Ancient Human (Neanderthal Genome - Altai) - Run ERR229911 (High Coverage)
Save-AncestralSample -Url "https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR229/ERR229911/ERR229911_1.fastq.gz" `
    -OutputPath "$destDir/ERR229911_1.fastq.gz" `
    -Description "Run ERR229911 (Neanderthal Genome) - Read 1" `
    -ReadIndex 0

Save-AncestralSample -Url "https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR229/ERR229911/ERR229911_2.fastq.gz" `
    -OutputPath "$destDir/ERR229911_2.fastq.gz" `
    -Description "Run ERR229911 (Neanderthal Genome) - Read 2" `
    -ReadIndex 1

Write-Host "All verifiable files processed."
Write-Host "You can now verify these files correspond to the Source IDs in src/data_generation.py"
