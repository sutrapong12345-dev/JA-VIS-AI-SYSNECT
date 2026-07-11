# J.A.R.V.I.S. v10.1 — SYSNECT AI Agent

ผู้ช่วย AI สำหรับงานส่วนตัวและองค์กร รองรับหลายโมเดล, streaming chat,
ฐานความรู้พร้อม citation, RBAC, reminders, structured tools และ audit log
แบบตรวจสอบการแก้ไขย้อนหลังได้

หน้าเว็บมีศูนย์ควบคุมบทสนทนา, reconnect, session reset, responsive layout,
Admin dashboard และ AI Quality Gate สำหรับคัดกรองข้อมูลก่อน fine-tune

## เอกสาร

- [คู่มือผู้ใช้งาน](docs/USER_MANUAL_TH.md)
- [คู่มือผู้ดูแลระบบ](docs/ADMIN_MANUAL_TH.md)
- [Security Foundation](SECURITY_FOUNDATION.md)
- [Training และ AI Quality](training/README.md)

## เริ่มใช้งานสำหรับนักพัฒนา

1. คัดลอก `backend/.env.example` เป็น `backend/.env`
2. ใส่ API key เฉพาะใน `.env` และห้าม commit
3. รัน `run_backend.bat`
4. รัน `run_frontend.bat`
5. เปิดหน้าเว็บตาม URL ที่ frontend server แสดง

ระบบควบคุมเครื่องและ Shell ปิดเป็นค่าเริ่มต้น ไม่ควรเปิดในระบบ production
จนกว่าจะกำหนด allowlist, approval policy และ isolation ครบถ้วน

## ทดสอบ

```powershell
venv\Scripts\python.exe -m unittest discover -s tests -v
```
