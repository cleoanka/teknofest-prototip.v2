# Codex CLI — AURA Kod İnceleme & İkinci-Görüş Rehberi

> 🔴 **GÜNCEL DURUM (18.06.2026) — DÜRÜST NOT:** Codex CLI bu oturumlarda **çalışmadı**:
> salt-okunur danışma çağrıları **0-çıktı ile takıldı / yanıt vermedi** (bkz. `gozlem.md` §7,
> `plan1.md` §6). Aşağıdaki rehber kurulum/kullanım için **referans** olarak korunur, ancak
> şu an **fiilen kullanılamıyor**; bu projedeki kararlar ve incelemeler Claude (Opus) tarafından
> yapıldı. Codex tekrar yanıt verir hale gelirse rehber olduğu gibi geçerlidir.
>
> **Bu belge ne?** Bu repoda **Codex CLI**'yi (OpenAI/ChatGPT kod ajanı) ne zaman ve nasıl
> kullanacağımızın rehberi. Codex CLI **v0.140.0** kurulu (`~/.local/bin/codex`) ve
> **ChatGPT ile login** (auth çalışıyor: `codex login status` → "Logged in using ChatGPT").
> Varsayılan model **`gpt-5.5`** (openai; `-m` ile değiştirilir). Headless doğrulandı.
>
> **Gemini vs Codex — iş bölümü:**
> - **Gemini** (`gemini.md`): web/güncel-bilgi araştırması, dış kaynak tarama, olgu ikinci-görüşü.
> - **Codex**: **KOD** işleri — bağımsız kod incelemesi, alternatif uygulama önerisi, kod
>   ikinci-görüşü, headless kod düzenleme (sandbox'lı). İkisi de **çapraz-kontrol** ister
>   (başka bir ajan; çıktısını çalışan test/koşumla doğrula).

## Headless (otomasyon) kullanımı
```bash
codex exec -s read-only "soru/inceleme"           # SALT-OKUNUR (güvenli; ikinci-görüş/analiz)
codex exec -s read-only --skip-git-repo-check "..." # repo dışı da çalışır
codex exec -m <model> -s read-only "..."           # model seç (vars: hesap default'u)
cat diff.patch | codex exec -s read-only "bu diff'i incele"   # stdin <stdin> bloğu olarak eklenir
codex exec --output-schema schema.json -s read-only "..."     # yapılandırılmış JSON çıktı
```
**Sandbox modları (`-s`):** `read-only` (varsayılan tercihimiz — analiz/inceleme) ·
`workspace-write` (dosya düzenleyebilir — dikkatli) · `danger-full-access` (KULLANMA, gerekmedikçe).

## Kod incelemesi (Codex'in en güçlü yanı)
```bash
codex review                       # mevcut repo/diff'i incele (etkileşimli)
codex exec review                  # headless kod incelemesi
codex exec -s read-only "PR/diff'imdeki olası bug, regresyon, kenar durumlarını eleştir"
```
> AURA'da kullanım: büyük değişikliklerden sonra **bağımsız ikinci-göz** olarak `codex review`
> çalıştır; bulguları kendi yargımla + testlerle süz (körü körüne uygulama).

## Diğer komutlar
- `codex apply` (`a`) — Codex ajanının ürettiği son diff'i `git apply` ile çalışma ağacına uygula.
- `codex doctor` — kurulum/auth/runtime sağlık teşhisi.
- `codex mcp` / `plugin` — MCP sunucu + eklenti yönetimi.
- `codex resume` / `fork` — önceki oturumu sürdür/çatalla. `codex sandbox` — sandbox'ta komut.
- `-c key=value` (TOML config override, ör. `-c model="..."`), `-p <profile>`, `--ephemeral`
  (oturum diske yazılmaz), `--ignore-user-config`.

## AURA'da kullanım desenleri
```bash
# Büyük commit/PR sonrası bağımsız kod incelemesi:
codex exec -s read-only "aura/plate/ ve aura/driver_state/ değişikliklerinde correctness
  bug'ı, regresyon, kenar durumu var mı? Somut dosya:satır ver."

# Alternatif uygulama / ikinci-görüş:
codex exec -s read-only "@aura/plate/normalize.py plaka oy-mantığını incele; pozisyon-veto +
  zemin koşulu yaklaşımına alternatif/iyileştirme öner."

# Hedefli düzeltme (dikkatli, sandbox'lı — sonra test + git diff incele):
codex exec -s workspace-write "X testini kıran Y'yi düzelt; başka davranışı değiştirme."
```
**Her zaman:** Codex'in önerisini/düzeltmesini `pytest -m "not integration"` + `ruff`/`black` +
`git diff` ile doğrula. Birincil değişiklikleri ben (Claude) yönetirim; Codex bağımsız
ikinci-görüş/inceleme + gerektiğinde headless kod yardımı içindir.
