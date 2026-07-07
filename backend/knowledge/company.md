# ฐานความรู้บริษัท SYSNECT

> ไฟล์นี้คือ "สมอง" ของผู้ช่วย AI สำหรับตอบคำถามเกี่ยวกับบริษัท SYSNECT
> แก้ไข/เพิ่มข้อมูลได้ตลอด แล้วเรียก POST /api/knowledge/reload (หรือรีสตาร์ท backend)
> ช่องที่เขียนว่า [โปรดเติม] คือข้อมูลที่ยังไม่ได้ยืนยัน — เติมให้ครบเพื่อให้ AI ตอบได้ถูกต้อง

---

## 1. ข้อมูลบริษัท (Company Profile)
- **ชื่อบริษัท:** SYSNECT
- **ชื่อเต็มตามกฎหมาย:** [โปรดเติม เช่น บริษัท ซิสเน็ค จำกัด]
- **ประเภทธุรกิจ:** บริการด้านเทคโนโลยีสารสนเทศ (IT Services / IT Managed Services / IT Service Management)
- **เว็บไซต์/โดเมนหลัก:** sysnect.co.th
- **ที่อยู่:** [โปรดเติม]
- **โทรศัพท์:** [โปรดเติม]
- **อีเมลติดต่อ:** [โปรดเติม]
- **เวลาทำการ:** [โปรดเติม]
- **ปีที่ก่อตั้ง / จำนวนพนักงาน:** [โปรดเติม]

## 2. บริการและสิ่งที่บริษัททำ (Services)
บริษัทให้บริการและดูแลงานด้าน IT ต่อไปนี้ (อ้างอิงจากระบบและงานที่มีจริง):
- **IT Service Desk / Ticket Management** — รับแจ้งปัญหาและติดตามงานผ่านระบบ GLPI
- **IT Asset Management** — จัดการทะเบียนทรัพย์สิน IT และตรวจสอบ Serial (IT Asset Inventory, Serial Comparison & Audit)
- **Network / Infrastructure** — งานเครือข่าย (ทีมมีพื้นฐาน CCNA)
- **Workflow Automation** — ระบบอัตโนมัติด้วย n8n (สรุป ticket รายวัน, ส่งอีเมลอัตโนมัติ)
- **Dashboard & Reporting** — แดชบอร์ดสรุปสถานะงานแบบเรียลไทม์ พร้อม export Excel/CSV/PDF
- **AI / Automation Solutions** — พัฒนาระบบผู้ช่วย AI ภายในองค์กร
- บริการเพิ่มเติมสำหรับลูกค้าภายนอก: [โปรดเติม]

## 3. ระบบและโครงสร้างพื้นฐาน (Systems & Infrastructure)
| ระบบ | รายละเอียด | ที่อยู่ / พอร์ต |
|---|---|---|
| GLPI IT Service Desk | ระบบ ticket หลัก (แหล่งข้อมูลต้นทาง) | itservicedesk.sysnect.co.th (apirest.php) |
| SSO / Authentication | ระบบล็อกอินรวม | auth.sysnect.co.th |
| Ticket Dashboard | เว็บแดชบอร์ดสรุป ticket | GitHub Pages (เว็บจริง) + Railway (API) |
| n8n (Primary backend) | Workflow automation / ETL / API หลัก | พอร์ต 5678, `/webhook/daily-tickets` |
| Node.js + Express (Fallback) | API สำรอง (Disaster Recovery) สลับอัตโนมัติเมื่อ n8n ช้าเกิน ~45 วินาที | พอร์ต 3000 |
| PostgreSQL 15 | ฐานข้อมูลกลาง (Single Source of Truth) | พอร์ต 5432, volume `sysnect_postgres_data` |
| Docker | รันทุก service แบบ microservices | — |

## 4. กระบวนการทำงานของ Ticket (Ticket Workflow)
- **5 สถานะ:** NEW → ASSIGNED → PENDING → SOLVED → CLOSED
- **ระดับความสำคัญ (Priority):** Critical, High, Medium, Low
- **SLA:** ค่าที่ใช้ในแดชบอร์ดตอนนี้เป็นค่าชั่วคราว (ตอบสนอง ~10 นาที / ปิดงาน ~7 วัน) — นโยบาย SLA จริง: [โปรดยืนยัน/เติม]
- **การแจ้งเตือน:** เมนูกระดิ่งแสดง ticket ที่เข้ามาใหม่ (NEW) และรายการใบเข้าใหม่ใน 24 ชั่วโมง
- **รายงาน:** สรุป Daily Ticket Summary ส่งอีเมลอัตโนมัติ พร้อมกราฟ (QuickChart)

## 5. สแต็กเทคโนโลยี (Technology Stack)
- **Backend/Automation:** n8n, Node.js + Express, PostgreSQL 15, Docker
- **Frontend:** Vanilla JavaScript, HTML5, CSS3 (ไม่ใช้ framework หนัก) — เน้นโหลดเร็ว, UI ระดับ Enterprise
- **Visualization/Export:** Chart.js (Doughnut chart), QuickChart, SheetJS (xlsx), html2pdf.js, DOMPurify
- **Deploy:** GitHub Pages (เว็บ), Railway (API)
- **จุดเด่นสถาปัตยกรรม:** Failover/Zero-downtime, Chunk rendering (วาดทีละ 50 รายการ), Data sanitization

## 6. โปรเจกต์ภายใน (Internal Projects)
- **SYSNECT Enterprise Ticket Dashboard** — แดชบอร์ดหลักที่ใช้งานจริง (data flow: GLPI → n8n → เว็บ)
- **Local AI Chatbot** — ผู้ช่วย AI แบบ local/ออฟไลน์ (Dify + Ollama บน Docker) — สถานะ: กำลังพัฒนา
- **J.A.R.V.I.S. AI Agent** — ผู้ช่วย AI ประจำองค์กรตัวนี้ (โปรเจกต์ปัจจุบัน)
- **เอกสารงาน:** แบบฟอร์ม CAR (Corrective Action Report), แบบบันทึกการปฏิบัติงาน

## 7. ทีมงาน / ผู้ติดต่อ (Team & Contacts)
- **ผู้ดูแลระบบ / IT Admin:** [โปรดเติม ชื่อ + ช่องทางติดต่อ]
- **ทีมพัฒนา:** [โปรดเติม]
- **ช่องทางแจ้งปัญหา IT:** ผ่านระบบ GLPI (itservicedesk.sysnect.co.th) — ช่องทางอื่น: [โปรดเติม]

## 8. คำถามที่พบบ่อย (FAQ)
**Q: จะแจ้งปัญหา IT อย่างไร?**
A: แจ้งผ่านระบบ GLPI ที่ itservicedesk.sysnect.co.th ระบบจะออกเลข ticket ให้ติดตามสถานะได้

**Q: ticket มีสถานะอะไรบ้าง?**
A: NEW, ASSIGNED, PENDING, SOLVED, CLOSED

**Q: ดูสรุปสถานะงานได้ที่ไหน?**
A: ที่ SYSNECT Ticket Dashboard (มีกราฟสรุป + filter + export Excel/CSV/PDF)

**Q: [เพิ่มคำถามที่พนักงานถามบ่อย]**
A: [โปรดเติม]
