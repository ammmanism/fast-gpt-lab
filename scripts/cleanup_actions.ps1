# Cleanup Script: Delete all failed GitHub Actions workflow runs
# This removes historical failures that can never pass (old commits)

Write-Host "🧹 Fetching all failed workflow runs..." -ForegroundColor Cyan

$failedRuns = gh run list --repo ammmanism/fast-gpt-lab --status failure --limit 200 --json databaseId | ConvertFrom-Json

$total = $failedRuns.Count
Write-Host "Found $total failed runs to delete." -ForegroundColor Yellow

$count = 0
foreach ($run in $failedRuns) {
    $count++
    $id = $run.databaseId
    Write-Host "  [$count/$total] Deleting run $id..." -NoNewline
    gh run delete $id --repo ammmanism/fast-gpt-lab 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host " ✅" -ForegroundColor Green
    } else {
        Write-Host " ⚠️ skipped" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "🟢 Cleanup complete! $count failed runs processed." -ForegroundColor Green
Write-Host "Your Actions tab should now be clean." -ForegroundColor Cyan
