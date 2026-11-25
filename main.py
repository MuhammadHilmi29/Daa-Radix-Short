import pandas as pd
from adaptive_hybrid_sort import adaptive_hybrid_sort

# 1. BACA FILE EXCEL
df = pd.read_excel("Untitled spreadsheet (3).xlsx")
df.columns = df.columns.str.strip()  # Hapus spasi di nama kolom ('Job ', 'skill ')

# Urutan level pendidikan untuk perbandingan
level_pendidikan = {"SMA": 1, "SMK": 1, "D3": 2, "D4": 3, "S1": 3, "S2": 4, "S3": 5}

#INPUT DATA PENGGUNA
print("CARI LOWONGAN KERJA")
job_dicari = input("Pekerjaan yang diinginkan (cth: Full Stuck / Data Analyst): ").strip().lower()
kota_dicari = input("Kota yang diinginkan (kosongkan jika bebas)               : ").strip().lower()
gaji_min = int(input("Gaji minimal (cth: 6000000, isi 0 jika bebas)             : ") or 0)
pendidikan = input("Pendidikan terakhir (SMA/D3/D4/S1/S2)                     : ").strip().upper()
pengalaman = int(input("Pengalaman kerja (tahun, cth: 2)                          : ") or 0)
skill_input = input("Skill yang dimiliki (pisahkan dengan koma, cth: Python, MySQL)   : ").lower()

skill_user = []
for s in skill_input.split(","):
    if s.strip():
        skill_user.append(s.strip())

#FILTER & HITUNG SKOR KECOCOKAN (0 - 100)
hasil_lowongan = []

for _, row in df.iterrows():
    job_lowongan = str(row["Job"]).strip()
    kota_lowongan = str(row["City"]).strip()
    gaji_lowongan = int(row["Salary"])
    edu_lowongan = str(row["Tingkat pendidikan"]).strip().upper()
    exp_lowongan = int(row["Pengalaman (Tahun)"])
    skill_lowongan = str(row["skill"]).lower()

    #kondisi kalau user isi job/kota/gaji, harus cocok
    if job_dicari and job_dicari not in job_lowongan.lower():
        continue
    if kota_dicari and kota_dicari != kota_lowongan.lower():
        continue
    if gaji_lowongan < gaji_min:
        continue

    # Hitung skor (Total maksimal 100)
    skor = 0

    # a. Skor Skill (maksimal 50 poin)
    cocok = 0
    for s in skill_user:
        if s in skill_lowongan:
            cocok += 1
    if skill_user:
        skor += int((cocok / len(skill_user)) * 50)

    # b. Skor Pendidikan (25 poin jika memenuhi)
    if level_pendidikan.get(pendidikan, 0) >= level_pendidikan.get(edu_lowongan, 0):
        skor += 25

    # c. Skor Pengalaman (25 poin jika memenuhi)
    if pengalaman >= exp_lowongan:
        skor += 25

    hasil_lowongan.append({
        "Perusahaan": row["Company"],
        "Job": job_lowongan,
        "Kota": kota_lowongan,
        "Gaji": gaji_lowongan,
        "Pendidikan": edu_lowongan,
        "Pengalaman": exp_lowongan,
        "Skill": row["skill"],
        "Skor": skor
    })

# URUTKAN PAKAI ADAPTIVE HYBRID SORT
daftar_skor = []
for item in hasil_lowongan:
    daftar_skor.append(item["Skor"])
skor_terurut = adaptive_hybrid_sort(daftar_skor)  # Hasil dari kecil ke besar
skor_terurut.reverse()                            # Dibalik jadi besar ke kecil

# Cocokkan kembali skor yang sudah urut dengan data lowongannya
hasil_final = []
for s in skor_terurut:
    for item in hasil_lowongan:
        if item["Skor"] == s and item not in hasil_final:
            hasil_final.append(item)
            break

# TAMPILKAN HASIL
print("\nHASIL REKOMENDASI LOWONGAN")
if not hasil_final:
    print("Tidak ada lowongan yang cocok.")
else:
    i = 1
    for item in hasil_final:
        print(f"{i}. {item['Perusahaan']} - {item['Job']} (Skor: {item['Skor']}%)")
        print(f"   Kota: {item['Kota']} | Gaji: Rp {item['Gaji']:,} | Min: {item['Pendidikan']}, {item['Pengalaman']} thn")
        print(f"   Skill: {item['Skill']}\n")
        i += 1