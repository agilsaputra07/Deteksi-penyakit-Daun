document.getElementById('fileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('preview').src = e.target.result;
            document.getElementById('preview-area').style.display = 'block';
        }
        reader.readAsDataURL(file);
    }
});

function deteksi() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (!file) {
        alert('Pilih foto daun terlebih dahulu!');
        return;
    }

    // Tampilkan loading
    document.getElementById('hasil').style.display = 'block';
    document.getElementById('hasilText').innerHTML = '⏳ Sedang menganalisis gambar...';

    const formData = new FormData();
    formData.append('file', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('hasilText').innerHTML = '❌ Error: ' + data.error;
        } else {
            document.getElementById('hasilText').innerHTML = `
                <table style="width:100%; text-align:left; border-collapse:collapse;">
                    <tr>
                        <td style="padding:8px;">🌿 <b>Penyakit</b></td>
                        <td style="padding:8px;"><b>${data.penyakit}</b></td>
                    </tr>
                    <tr style="background:#f1f8e9;">
                        <td style="padding:8px;">📊 <b>Tingkat Keyakinan</b></td>
                        <td style="padding:8px;"><b>${data.confidence}%</b></td>
                    </tr>
                    <tr>
                        <td style="padding:8px;">🎨 <b>Mean Hue (Warna)</b></td>
                        <td style="padding:8px;">${data.fitur_warna.mean_hue}</td>
                    </tr>
                    <tr style="background:#f1f8e9;">
                        <td style="padding:8px;">🎨 <b>Mean Saturation</b></td>
                        <td style="padding:8px;">${data.fitur_warna.mean_saturation}</td>
                    </tr>
                    <tr>
                        <td style="padding:8px;">📐 <b>Luas Daun</b></td>
                        <td style="padding:8px;">${data.fitur_morfologi.luas} px²</td>
                    </tr>
                    <tr style="background:#f1f8e9;">
                        <td style="padding:8px;">📐 <b>Keliling Daun</b></td>
                        <td style="padding:8px;">${data.fitur_morfologi.keliling} px</td>
                    </tr>
                </table>
            `;
        }
    })
    .catch(error => {
        document.getElementById('hasilText').innerHTML = '❌ Terjadi kesalahan, coba lagi.';
    });
}