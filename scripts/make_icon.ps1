Add-Type -AssemblyName System.Drawing

$dir = "d:\vt_application_version_1\Attendance_Application\assets"
if (-not (Test-Path $dir)) {
    New-Item -ItemType Directory -Path $dir | Out-Null
}

$size = 256
$bmp = New-Object System.Drawing.Bitmap $size, $size
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.Clear([System.Drawing.Color]::Transparent)

# Colors
$bgColor = [System.Drawing.Color]::FromArgb(255, 18, 27, 45)
$tealColor = [System.Drawing.Color]::FromArgb(255, 21, 155, 146)
$accentColor = [System.Drawing.Color]::FromArgb(255, 26, 188, 156)
$greenColor = [System.Drawing.Color]::FromArgb(255, 16, 185, 129)
$whiteColor = [System.Drawing.Color]::FromArgb(255, 240, 246, 252)
$darkTeal = [System.Drawing.Color]::FromArgb(255, 26, 37, 58)

$bgBrush = New-Object System.Drawing.SolidBrush $bgColor
$tealPen = New-Object System.Drawing.Pen $tealColor, 10
$greenPen = New-Object System.Drawing.Pen $greenColor, 8
$whiteBrush = New-Object System.Drawing.SolidBrush $whiteColor
$tealBrush = New-Object System.Drawing.SolidBrush $tealColor
$darkBrush = New-Object System.Drawing.SolidBrush $darkTeal

# Outer Body Circle
$g.FillEllipse($bgBrush, 12, 12, 232, 232)
$g.DrawEllipse($tealPen, 12, 12, 232, 232)

# Lens Ring
$g.FillEllipse($darkBrush, 64, 64, 128, 128)
$g.DrawEllipse($tealPen, 64, 64, 128, 128)
$g.FillEllipse($tealBrush, 96, 96, 64, 64)
$g.FillEllipse($whiteBrush, 116, 116, 24, 24)

# Biometric Scanner Brackets
$p1 = New-Object System.Drawing.Point 44, 72
$p2 = New-Object System.Drawing.Point 44, 44
$p3 = New-Object System.Drawing.Point 72, 44
$g.DrawLines($greenPen, [System.Drawing.Point[]]@($p1, $p2, $p3))

$p4 = New-Object System.Drawing.Point 212, 72
$p5 = New-Object System.Drawing.Point 212, 44
$p6 = New-Object System.Drawing.Point 184, 44
$g.DrawLines($greenPen, [System.Drawing.Point[]]@($p4, $p5, $p6))

$p7 = New-Object System.Drawing.Point 44, 184
$p8 = New-Object System.Drawing.Point 44, 212
$p9 = New-Object System.Drawing.Point 72, 212
$g.DrawLines($greenPen, [System.Drawing.Point[]]@($p7, $p8, $p9))

$p10 = New-Object System.Drawing.Point 212, 184
$p11 = New-Object System.Drawing.Point 212, 212
$p12 = New-Object System.Drawing.Point 184, 212
$g.DrawLines($greenPen, [System.Drawing.Point[]]@($p10, $p11, $p12))

$pngPath = Join-Path $dir "icon.png"
$icoPath = Join-Path $dir "icon.ico"

$bmp.Save($pngPath, [System.Drawing.Imaging.ImageFormat]::Png)

$hIcon = $bmp.GetHicon()
$icon = [System.Drawing.Icon]::FromHandle($hIcon)
$fileStream = New-Object System.IO.FileStream($icoPath, [System.IO.FileMode]::Create)
$icon.Save($fileStream)
$fileStream.Close()

$g.Dispose()
$bmp.Dispose()
Write-Host "Created assets/icon.png and assets/icon.ico successfully!"
