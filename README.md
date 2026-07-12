# J.A.R.V.I.S. v10.5 — SYSNECT AI Agent

ผู้ช่วย AI สำหรับงานส่วนตัวและองค์กร รองรับหลายโมเดล, streaming chat,
ฐานความรู้พร้อม citation, RBAC, reminders, structured tools และ audit log
แบบตรวจสอบการแก้ไขย้อนหลังได้

หน้าเว็บมีศูนย์ควบคุมบทสนทนา, reconnect, session reset, responsive layout,
Admin dashboard และ AI Quality Gate สำหรับคัดกรองข้อมูลก่อน fine-tune

v10.2 เพิ่ม JARVIS Holographic Activity Projector แบบ Canvas 3D ที่แสดงงาน
จริงตาม stage จาก backend, ช่องพิมพ์หลายบรรทัด, Secure Confirmation Modal,
Request ID และ Provider Circuit Breaker สำหรับ fallback ที่เร็วและตรวจสอบได้

v10.3 ยกระดับ Hologram เป็น Cinematic Work Console แสดง pipeline แบบ
`INPUT → CONTEXT → AI CORE → ACTION → OUTPUT`, ภาพเคลื่อนไหวเฉพาะงานค้นหา,
เสียง, เอกสาร, เครื่องมือ, ผลสำเร็จและข้อผิดพลาด พร้อมโหมดขยายเต็มพื้นที่
และรองรับมือถือ/Reduced Motion

v10.4 เพิ่ม Interactive Project Workspace ที่มีแท่นฉายและลำแสง, floating
intelligence panels, live event stream และการควบคุมวัตถุด้วย mouse/touch,
wheel, ปุ่มซูมและคีย์บอร์ด แนวคิดการโต้ตอบได้รับแรงบันดาลใจจากงาน 3D hologram
สมัยใหม่ แต่โค้ดและภาพฉายทั้งหมดของ SYSNECT สร้างขึ้นใหม่ภายในโปรเจกต์นี้

v10.5 เพิ่ม AI Freshness Foundation: คำถามวันและเวลาตอบจากนาฬิกา Backend
เขต `Asia/Bangkok` โดยไม่ให้โมเดลเดา, ส่งเวลาแบบมี timezone เข้า context ทุกคำขอ,
กำหนดนโยบายตรวจข้อมูลที่เปลี่ยนแปลงได้ และเพิ่ม curated training examples
สำหรับป้องกันวันที่เก่า ข่าวแต่ง และข้อมูลปัจจุบันที่ยังไม่ผ่านการตรวจสอบ

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
