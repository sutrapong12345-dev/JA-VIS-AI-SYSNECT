# คู่มือผู้ดูแลระบบ J.A.R.V.I.S. — SYSNECT

## 1. สถาปัตยกรรมความปลอดภัย

```text
ผู้ใช้ → Cloudflare Access → Named Tunnel → FastAPI (127.0.0.1:8000)
                                      ├─ AI providers
                                      ├─ RAG database
                                      ├─ Append-only audit database
                                      └─ Restricted tool worker
```

Backend ต้อง bind ที่ `127.0.0.1` เท่านั้น ไม่ควรเปิดพอร์ต 8000 ต่อ Internet
โดยตรง

## 2. ไฟล์ Configuration

ใช้ `backend/.env.example` เป็นต้นแบบ แล้วเก็บค่าจริงใน `backend/.env`
ซึ่งถูก `.gitignore` ไว้แล้ว

ค่าความปลอดภัยสำคัญ:

```text
CORS_ORIGINS=https://ai.example.com
SESSION_TOKEN_TTL_SECONDS=43200
ADMIN_SESSION_TTL_SECONDS=28800
ENABLE_COMMAND_EXECUTION=false
ENABLE_SHELL_COMMANDS=false
REQUIRE_SHELL_CONFIRMATION=true
TRUST_CLOUDFLARE_ACCESS_HEADERS=false
REQUIRE_ORG_IDENTITY=false
```

ห้ามเปิดสองค่าของ Cloudflare Access จนกว่า hostname จะถูกป้องกันด้วย Access
และ backend รับ traffic ผ่าน local `cloudflared` เท่านั้น

## 3. Named Cloudflare Tunnel

เครื่องปัจจุบันยังไม่มี `cert.pem` จึงต้อง login ด้วยบัญชี Cloudflare ก่อน:

```powershell
cloudflared tunnel login
cloudflared tunnel create jarvis-sysnect
```

สร้าง `%USERPROFILE%\.cloudflared\config.yml`:

```yaml
tunnel: <TUNNEL-UUID>
credentials-file: C:\Users\<USER>\.cloudflared\<TUNNEL-UUID>.json
ingress:
  - hostname: ai.example.com
    service: http://127.0.0.1:8000
  - service: http_status:404
```

สร้าง DNS route:

```powershell
cloudflared tunnel route dns jarvis-sysnect ai.example.com
cloudflared tunnel run jarvis-sysnect
```

DNS record และ Tunnel เป็นคนละส่วน หาก Tunnel หยุด DNS จะยังอยู่และผู้ใช้จะเห็น
Cloudflare error จึงต้องติดตั้ง service/watchdog สำหรับ Named Tunnel

เอกสารทางการ:

- https://developers.cloudflare.com/tunnel/routing/
- https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/

## 4. Cloudflare Access และ SSO

ใน Cloudflare Zero Trust:

1. ไปที่ `Access controls > Applications`
2. สร้าง `Self-hosted` application
3. ระบุ hostname ของ J.A.R.V.I.S.
4. สร้าง Allow policy เฉพาะอีเมลองค์กรหรือ IdP group
5. ตั้ง session duration ตามนโยบายบริษัท
6. ทดสอบว่าผู้ใช้ที่ไม่ผ่าน policy เข้า hostname ไม่ได้

จากนั้นตั้งค่า:

```text
TRUST_CLOUDFLARE_ACCESS_HEADERS=true
REQUIRE_ORG_IDENTITY=true
ORG_ALLOWED_DOMAINS=sysnect.example
MANAGER_EMAILS=manager1@sysnect.example
ADMIN_EMAILS=admin1@sysnect.example
```

`ADMIN_EMAILS` กำหนดบทบาทองค์กร แต่ action ระดับเครื่องยังต้อง login ด้วยรหัสผ่าน
Admin ภายใน J.A.R.V.I.S. อีกชั้น

## 5. RBAC

ระดับสิทธิ์คือ `staff < manager < admin`

- Session รับ actor ID และ role จาก Access identity
- API ตรวจ token และ session ownership ก่อนทุกครั้ง
- Admin endpoints ตรวจ Admin Mode เพิ่ม
- RAG กรองเอกสารตาม role ก่อนส่งข้อมูลให้โมเดล

ตั้งสิทธิ์เอกสารด้วยชื่อไฟล์:

```text
staff__employee_handbook.md
manager__department_budget.md
admin__disaster_recovery.md
```

ไฟล์ที่ไม่มี prefix ถือเป็น `staff`

## 6. Knowledge RAG และ Citation

วาง `.md` หรือ `.txt` ใน `backend/knowledge` แล้วเรียก:

```text
POST /api/knowledge/reload
```

ระบบจะ:

1. คำนวณ hash เอกสาร
2. แบ่งเป็น chunk แบบมี overlap
3. บันทึก role ของเอกสาร
4. ค้นเฉพาะ chunk ที่เกี่ยวข้องกับคำถาม
5. กรองตามสิทธิ์ก่อนสร้าง prompt
6. กำหนด citation `[KB:filename#chunk-N]`

ฐานข้อมูลอยู่ใน `backend/data/knowledge.db` และไม่ถูก commit

## 7. Audit Log

Audit SQLite อยู่ที่ `backend/data/audit.db` มีคุณสมบัติ:

- append-only trigger ป้องกัน UPDATE/DELETE
- SHA-256 hash chain เชื่อมทุก event
- บันทึก actor, role, session, action, outcome, target และรายละเอียด

ตรวจสอบผ่าน Admin API:

```text
GET /api/audit/events?limit=100
GET /api/audit/verify
```

Production PostgreSQL ใช้ migration:

```text
backend/migrations/postgresql_audit.sql
```

หลัง migration ให้แยก database role ของ application และถอนสิทธิ์
`UPDATE`, `DELETE`, `TRUNCATE` จากตาราง audit

## 8. Structured Tools และ Worker

Tool definitions อยู่ใน `backend/agent_tools.py` ส่วน read-only worker อยู่ใน
`backend/sandbox_worker.py`

หลักการ:

- Tool ต้องมีชื่อและ schema ที่ลงทะเบียนไว้
- ตรวจ role และ argument ใน application code
- ไม่รับ generic command หรือ code
- read-only tools รันใน process แยกพร้อม timeout
- action ที่มีผลกระทบสร้าง action ID และรออนุมัติ
- raw Shell ปิดใน production

การเพิ่ม Tool ใหม่ต้องเพิ่ม test อย่างน้อย:

- role ที่อนุญาตและไม่อนุญาต
- argument ผิดประเภท/เกินขอบเขต
- approval requirement
- timeout/error handling
- audit event

## 9. การทดสอบก่อน Deploy

```powershell
venv\Scripts\python.exe -m py_compile backend\main.py backend\*.py
venv\Scripts\python.exe -m unittest discover -s tests -v
```

ตรวจเพิ่มด้วยตนเอง:

- ไม่มี token ต้องได้ 401
- token ข้าม Session ต้องได้ 403
- Staff เข้า audit/system log ต้องได้ 403
- เอกสาร Admin ค้นด้วย Staff ไม่พบ
- Shell ไม่ทำงาน
- Audit chain แสดง `valid: true`
- CORS ยอมเฉพาะ frontend origin

## 10. AI Quality Gate และ Fine-tuning

แผง Admin ส่วน `AI QUALITY` ใช้ตรวจข้อมูลก่อน fine-tune:

1. กดตรวจคุณภาพชุดข้อมูลปัจจุบัน
2. อ่านคะแนน จำนวนผ่าน/คัดออก และเหตุผลหลัก
3. กดสร้างชุดฝึกใหม่ผ่าน Quality Gate
4. ตรวจว่า train/validation ถูกสร้างและ readiness ผ่านเกณฑ์

เกณฑ์มาตรฐานคือ ตัวอย่างผ่านอย่างน้อย 100 คู่ คะแนนเฉลี่ย 80/100 และอัตราผ่าน 80%
ระบบจะไม่เริ่ม GPU training เองจากหน้าเว็บ เพื่อป้องกันงานที่ใช้ทรัพยากรสูงโดยไม่ตั้งใจ

ตรวจจาก command line:

```powershell
venv\Scripts\python.exe finetune.py --check-only
```

ดูรายละเอียดทั้งหมดใน `training/README.md` ชุดข้อมูลที่สร้างจาก chat logs และรายงานคุณภาพถูก ignore จาก Git
และต้องอยู่ในเครื่องหรือพื้นที่ข้อมูลที่องค์กรอนุมัติเท่านั้น

## 11. Coordinated Deployment

Frontend และ backend ต้อง deploy พร้อมกัน:

1. สำรอง `.env`, `logs` และ `backend/data`
2. รัน test
3. Commit เฉพาะ source/docs ห้าม commit secret หรือข้อมูลจริง
4. Push เพื่อให้ GitHub Pages deploy frontend
5. รอ GitHub Pages สำเร็จ
6. Restart backend
7. เปิดหน้าเว็บใหม่และตรวจ Secure Session
8. ตรวจ `/api/audit/verify`

หาก frontend เก่ายัง cache อยู่ให้กด `Ctrl+F5`

## 12. Backup และ Recovery

สำรองอย่างน้อย:

- `backend/.env` ไป secret vault
- `backend/knowledge` ตาม classification
- `backend/data/audit.db`
- `backend/data/knowledge.db` หรือสร้างใหม่จากเอกสารได้
- `logs` ตาม retention policy

หากเกิดเหตุผิดปกติ:

1. กด Security Lockdown
2. หยุด Named Tunnel
3. เก็บ audit/log แบบ read-only
4. หมุน API keys และ Admin password เมื่อสงสัยว่ารั่ว
5. ตรวจ hash chain และ event timeline
6. แก้ไขใน branch ใหม่และ deploy แบบ coordinated
7. ห้ามลบ audit record เพื่อซ่อนเหตุการณ์

## 13. ข้อห้าม Production

- ห้ามใช้ Quick Tunnel เป็นช่องทางถาวร
- ห้ามใช้ `CORS_ORIGINS=*`
- ห้ามเปิด `ENABLE_SHELL_COMMANDS=true`
- ห้ามเชื่อ Cloudflare identity header หาก hostname ไม่ได้อยู่หลัง Access
- ห้าม commit `.env`, database, logs, chat history, training dataset หรือข้อมูลจริง
- ห้าม fine-tune จาก raw chat logs โดยไม่มีการตรวจและลบข้อมูลส่วนบุคคล
