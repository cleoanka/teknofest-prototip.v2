# Gemini CLI — AURA Geliştirme & Araştırma Rehberi

> 🟡 **GÜNCEL DURUM (18.06.2026) — DÜRÜST NOT:** Gemini CLI **kısmen** çalışıyor.
> Aşağıda ⭐ önerilen `gemini-3.1-pro-preview` (ve diğer 3.x pro modelleri) bu hesapta
> **`403 SUBSCRIPTION_REQUIRED`** veriyor (enterprise Code-Assist lisansı istiyor).
> **Çalışan model: `gemini-2.5-flash`** — mobil-RN ve final/CAMARA araştırması bununla yapıldı
> (bkz. `gozlem.md` §7, `plan1.md` §6). Yani: araştırma için Gemini'yi **`-m gemini-2.5-flash`
> ile** çağır; pro modelleri lisans gelene dek kullanma. Aşağıdaki model listesi kurulu
> sürümün sunduğu adları gösterir (hepsine erişim YOK — erişilebilirlik ≠ listede görünmek).
>
> **Bu belge ne?** Bu repoda **Gemini CLI**'yi (özellikle web-araştırması + ikinci-görüş
> için) sonuna kadar nasıl kullanacağımızın rehberi. Gemini CLI **v0.46.0** kurulu ve
> auth'lu (`which gemini` → `~/.npm-global/bin/gemini`). Cheat-sheet'in bir kısmı bizzat
> Gemini'ye (`gemini-3.1-pro-preview`) hazırlatıldı.
>
> ⚠️ **Altın kural (YOLO26 dersi):** Gemini'nin bilgisi güncel olmayabilir — bir keresinde
> "YOLO26 specs not yet public" dedi, oysa ortamımızda `ultralytics 8.4.66` + `yolo26l.pt`
> gerçekten çalışıyor. **Gemini'yi araştırma/ikinci-görüş için kullan, ama iddiaları her
> zaman çalışan ortama (kurulu paket, gerçek koşum) karşı çapraz-kontrol et.**

## Bu ortamdaki modeller (doğrulandı)
`-m <id>` ile seçilir; bu ortamda mevcut olanlar:
- **`gemini-3.1-pro-preview`** ⭐ — en güçlü akıl yürütme; **ciddi araştırma/sentez için bunu kullan**
  (alias `pro`). `gemini-3.1-pro-preview-customtools` da var.
- `gemini-3-pro-preview`, `gemini-3-flash-preview` (varsayılan aktif), `gemini-3.5-flash`,
  `gemini-3.1-flash-lite` — hızlı/ucuz işler (log ayrıştırma, basit özet).
- `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`.
- Aliaslar: `auto`, `pro`, `flash`, `flash-lite`, `auto-gemini-3`, `auto-gemini-2.5`.
- Bağlam penceresi: CLI varsayılan geçmiş penceresi ~150k token; pro modeller çok daha geniş
  bağlamı `@dosya` / `--include-directories` ile yutabilir.

## Headless (otomasyon) kullanımı — cheat-sheet

### Temel
```bash
gemini -p "soru"                         # tek-atış, etkileşimsiz (REPL'e girmez)
cat error.log | gemini -p "crash nedenini bul"   # stdin context olarak eklenir
gemini -m gemini-3.1-pro-preview -p "..." # PRO model (karmaşık akıl yürütme)
gemini -m flash -p "..."                  # hızlı/ucuz
gemini -p "..." -o json                   # final çıktı tek JSON (parse edilebilir)
gemini -p "..." -o stream-json            # JSONL akış (canlı ilerleme / jq)
```

### Bağlam enjeksiyonu (güçlü)
```bash
gemini -p "@config/default.yaml dosyasındaki dedektör eşiklerini incele ve öner"
gemini --include-directories aura/plate -p "plaka OCR hattını gözden geçir"
```
`@yol/dosya` belirli dosyaları, `--include-directories` tüm klasörü bağlama alır.

### Onay/güvenlik modları (otomasyonda kritik)
```bash
gemini -p "..." --approval-mode plan      # SALT-OKUNUR (araştırma/plan; güvenli — varsayılan tercihimiz)
gemini -p "..." --approval-mode auto_edit # düzenlemeleri oto-onayla, shell'i sor
gemini -y -p "..."                        # YOLO: tüm onayları atla (DİKKAT — yalnız güvenli işlerde)
gemini -s -p "..."                        # shell araçlarını sandbox'ta çalıştır
```
> AURA'da araştırma için **`--approval-mode plan`** (salt-okunur) tercih edilir; Gemini'nin
> repoda değişiklik yapmasını istemiyoruz (değişiklikleri biz yapay zekâ tarafında yönetiyoruz).

### Oturum / worktree / alt-komutlar
```bash
gemini -r latest -p "..."                 # son oturumu sürdür (-r 5 = indeksli)
gemini --session-id arastirma_plaka -p "..."   # paralel işlerde izolasyon
gemini --list-sessions
gemini -w -p "..."                        # yeni git worktree'de çalıştır
gemini mcp / extensions / skills / hooks  # MCP, eklenti, skill, lifecycle hook yönetimi
gemini --list-extensions
```

## Diğer özellikler (REPL/proje)
- **`GEMINI.md`** (proje kökü): `CLAUDE.md` gibi otomatik yüklenen proje-bağlam dosyası.
  İstenirse AURA için minimal bir `GEMINI.md` eklenip her gemini koşumu repo-farkında yapılabilir.
- **Özel subagent'lar:** `.gemini/agents/*.md` (YAML frontmatter) — ana ajan ilgili `description`'a
  göre otomatik delege eder; `@agent-adı <prompt>` ile manuel.
- **Plan Mode** (salt-okunur strateji onayı), **Rewind/Checkpoint** (REPL'de `Esc Esc`).
- **Policy** dosyaları `~/.gemini/policies/*.toml` (allow/deny/ask_user — araç güvenliği).
- Slash komutları (REPL): `/help`, `/settings`, `/agents list`, `/quit --delete` …

## AURA'da kullanım desenleri (örnekler)
```bash
# Karanlık/açılı plaka için perspektif düzeltme + OCR araştırması (PRO + salt-okunur):
gemini -m pro --approval-mode plan -p "TR license plate OCR on dark/angled CCTV: best open-source
  perspective-correction (dewarp) + OCR pipelines for low-light plates. Cite repos/papers."

# Eksik sınıflar için açık veri seti araştırması:
gemini -m pro --approval-mode plan -p "Open datasets (Roboflow/Kaggle) for cigarette, seatbelt,
  Turkish minibus detection (YOLO format). List with URLs + license + image counts."

# Resmi sayı çekme (sonra ortamla çapraz-kontrol et!):
gemini -m pro -p "Ultralytics YOLO11 official COCO val mAP50-95 table. Markdown + source URL."
```
**Her zaman:** sonucu `eval`/`model.val`/kurulu paket gibi yerel gerçeklerle teyit et.
