# Ustawienia
$ECR_REGISTRY = "714038743086.dkr.ecr.eu-central-1.amazonaws.com"
$IMAGE_NAME = "backend"
$Env:AWS_REGION = "eu-central-1"
$Env:AWS_DEFAULT_REGION = "eu-central-1"
$Env:AWS_ACCESS_KEY_ID = "ASIA2MQAGSAXHIO6BQ4Z"
$Env:AWS_SECRET_ACCESS_KEY = "8mPRBxEbEvXn1+TMSCnP3hhp12JzACSl+h/QyMRb"
$Env:AWS_SESSION_TOKEN = "IQoJb3JpZ2luX2VjEKH//////////wEaCXVzLWVhc3QtMSJGMEQCIAM1cP5QQ19CKE6GWL4maom9lA2WVkWhsEVqF98a91xeAiB8nXV7M9Aze2YLNsZjgZvzRD+VubRI6q+n7ZInkceDBSqZAgg6EAAaDDcxNDAzODc0MzA4NiIMc7pCVRm0pyU3cS44KvYB+Z4/t+20hGMVULrh1NlFBNb9Vtc7sHXlnE8Wh+xV26ZHRSt48oCxfbzwQE5Z7AiizAfSCJi3Tf9EsVHuq9Zc12jEWHflnZp3V9/Y13vw7GCidOlwIIotSUxNgxRfGUeTvNyNyjb8IoFF5ONM5Ih96RxxMNAThjpTBjP8CH4R2buBdXMInnZGXK8ZGWbev3++NODuEU0sCv0Pjd/2pFGKTnJhxJpzh9UJzc2WQT1q4SemE0pai5r4F9dXHIDaCAXQaXyYxQD8TuxI8CK0+6beWrj726HKC2mvlqI1/9u3cEK8DJhyRTep3Vmgp0kn97/p0MM1hjo0MPL85LkGOp4BNINIaYpY7Aca5Vzh20v5nHT/u/Ggp2WjJH0qDXg+3W3/D69bLAV4xaKUINqC2xsJHLTLgSg8b+p7EvpjttEDVGjr8AXX/SSUEp5NT20+tnq6A/Jc1TjIn33ivwNYtm3KEeEz4i/ZcQ/TL9LOYDLKlKvrmxJ3QkwK9VpTELMrFkgCiiBzV6Yt5KLn+kNadcaFjeyUypvVs54ot4LpG8A="

# Generowanie numeru wersji na podstawie daty i czasu
$VERSION = Get-Date -Format "yyyyMMdd.HHmm"

Write-Host "Rozpoczynam proces deploymentu..."
Write-Host "Numer wersji: $VERSION"

# 1. Logowanie do AWS ECR
Write-Host "Logowanie do AWS ECR..."
aws ecr get-login-password --region $Env:AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY
if ($LASTEXITCODE -ne 0) {
    Write-Host "Blad: Logowanie do ECR nie powiodlo sie"
    exit 1
}

# 2. Budowanie obrazu
Write-Host "Budowanie obrazu Dockera..."
docker build -t $IMAGE_NAME .
if ($LASTEXITCODE -ne 0) {
    Write-Host "Blad: Budowanie obrazu nie powiodlo sie"
    exit 1
}

# 3. Tagowanie obrazu
Write-Host "Tagowanie obrazu..."
docker tag "${IMAGE_NAME}:latest" "${ECR_REGISTRY}/${IMAGE_NAME}:${VERSION}"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Blad: Tagowanie obrazu nie powiodlo sie"
    exit 1
}

# 4. Push obrazu do ECR
Write-Host "Wysylanie obrazu do ECR..."
docker push "${ECR_REGISTRY}/${IMAGE_NAME}:${VERSION}"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Blad: Push obrazu nie powiodl sie"
    exit 1
}

Write-Host "Deployment zakonczony sukcesem!"
Write-Host "Wdrozona wersja: $VERSION"
Write-Host "Obraz dostepny pod: $ECR_REGISTRY/${IMAGE_NAME}:{$VERSION}"