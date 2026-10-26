// 시험 페이지 공통 (aging.html, post.html): API 호출, 서버 상태, 최근 결과 목록, 결과 그리기
const $ = id => document.getElementById(id);
const radio = name => document.querySelector(`input[name=${name}]:checked`).value;
const setRadio = (name, v) => { const el = document.querySelector(`input[name=${name}][value=${v}]`); if (el) el.checked = true; };
const KIND = { restore: '복원', aging: '나이 변환', hair: '머리', makeup: '화장', glasses: '안경' };

// ---------------------------------------------------------------- 저장 (브라우저에만, 없어도 동작)
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} },
};

// ---------------------------------------------------------------- API 키
function initApiKey() {
  $('apikey').value = store.get('apikey', '');
  $('apikey').addEventListener('change', () => store.set('apikey', $('apikey').value));
}
const headers = () => ($('apikey') && $('apikey').value) ? { 'X-API-Key': $('apikey').value } : {};

async function api(path, form) {
  const r = await fetch(path, { method: form ? 'POST' : 'GET', body: form, headers: headers() });
  let data = null;
  try { data = await r.json(); } catch (e) {}
  if (form && $('raw')) $('raw').textContent = `${r.status} ${path}\n` + JSON.stringify(data, null, 2);
  if (!r.ok) throw new Error(`${r.status}: ${(data && data.detail) || r.statusText}`);
  return data;
}

async function blobUrl(path) {  // API 키가 있으면 <img src> 로는 헤더를 못 붙여서 fetch 로 받는다
  const r = await fetch(path, { headers: headers() });
  if (!r.ok) throw new Error(`${r.status}`);
  return URL.createObjectURL(await r.blob());
}

// ---------------------------------------------------------------- 서버 상태
async function health() {
  try {
    const h = await api('/health');
    $('health').className = 'status ' + (h.busy ? '' : 'ok');
    $('health').textContent = `${h.gpu || 'GPU 없음'} · ${h.busy ? '작업 중' : '대기'}` + (h.loaded.length ? ` · 올라간 모델: ${h.loaded.join(', ')}` : '');
  } catch (e) {
    $('health').className = 'status err';
    $('health').textContent = '서버 연결 안 됨';
  }
}
function initHealth() { health(); setInterval(health, 5000); }

// ---------------------------------------------------------------- 진행 표시
function busy(prefix, on, text) {
  const btn = $(prefix + '-run'), st = $(prefix + '-status');
  btn.disabled = on;
  clearInterval(btn._t);
  if (on) {
    const t0 = Date.now();
    st.className = 'status';
    st.textContent = text;
    btn._t = setInterval(() => st.textContent = `${text} ${((Date.now() - t0) / 1000).toFixed(0)}초…`, 500);
  }
}
const done = (prefix, msg, ok = true) => { const st = $(prefix + '-status'); st.className = 'status ' + (ok ? 'ok' : 'err'); st.textContent = msg; };

// ---------------------------------------------------------------- 최근 결과 (두 페이지가 같이 쓴다)
const recent = () => store.get('recent', []);
function remember(res) {
  const item = { result_id: res.result_id, kind: res.kind, sex: res.sex, parent_id: res.parent_id,
                 keys: res.images.map(im => im.key), time: new Date().toLocaleTimeString() };
  store.set('recent', [item, ...recent().filter(r => r.result_id !== res.result_id)].slice(0, 20));
}
function forget(rid) { store.set('recent', recent().filter(r => r.result_id !== rid)); }
const label = r => `${KIND[r.kind] || r.kind} · ${r.keys ? r.keys.join(',') : ''} · ${r.result_id}`;

// ---------------------------------------------------------------- 결과 그리기
const imgLabel = im => im.label ? im.label : im.age != null ? `${im.age}세` : im.key.startsWith('age') ? `${im.key.slice(3)}세` : im.key === 'face' ? '사진' : im.key;

// 신원 점수(ArcFace)를 이미지 아래에. 복원 결과는 원본 대비, 복원 얼굴로 변환한 결과는 복원 대비·원본 대비 둘 다
function addScores(res) {
  for (const im of res.images) {
    const s = res.extra.identity?.[im.key]?.arcface, o = res.extra.identity_original?.[im.key]?.arcface;
    const name = res.kind === 'restore' ? 'ArcFace(원본 대비)' : o != null ? 'ArcFace(복원 대비)' : 'ArcFace';
    if (s != null) im.info = { ...im.info, [name]: s.toFixed(2) };
    if (o != null) im.info = { ...im.info, 'ArcFace(원본 대비)': o.toFixed(2) };
    if (res.kind === 'restore') im.label = im.key === 'input' ? '원본 (정렬)' : '복원';
  }
}

function figure(rid, im, extraButtons) {
  const fig = document.createElement('figure');
  const img = document.createElement('img');
  img.alt = im.key;
  fig.append(img);
  const cap = document.createElement('figcaption');
  const info = Object.entries(im.info || {}).map(([k, v]) => `${k} ${v}`).join(' · ');
  cap.innerHTML = `<b>${imgLabel(im)}</b><span>${info}</span>`;
  const a = document.createElement('a');
  a.textContent = 'PNG';
  a.download = `${rid}_${im.key}.png`;
  cap.append(a);
  if (extraButtons) cap.append(...extraButtons);
  fig.append(cap);
  blobUrl(im.url).then(u => { img.src = u; a.href = u; }).catch(e => cap.append(' 이미지 실패 ' + e.message));
  return fig;
}

function resultGrid(res, extraButtons) {
  const grid = document.createElement('div');
  grid.className = 'grid';
  for (const im of res.images) grid.append(figure(res.result_id, im, extraButtons && extraButtons(im)));
  return grid;
}

// ---------------------------------------------------------------- DB 저장 (origin 이 있는 결과만, ③ 작업 화면과 ② 후처리 화면)
const TTL_NOTE = '임시 결과는 24시간 뒤 지워집니다. DB 에 저장한 사진은 그대로 남습니다.';
const canSave = (res, im) => !!res.origin && im.key.startsWith('age');

function markSaved(btn, saved, created) {
  btn.className = 'small ok';
  btn.textContent = created === false ? '저장됨 (이미 저장된 결과)' : '저장 완료';
  btn.dataset.saved = saved.aiImageId;
}

// 결과 이미지 하나의 '선택한 결과 저장' 버튼. 저장 이력은 result_id 단위로 Java 에서 다시 읽어 표시한다.
function saveButton(res, im, onSaved) {
  if (!canSave(res, im)) return null;
  const b = document.createElement('button');
  b.className = 'small';
  b.textContent = '선택한 결과 저장';
  b.dataset.saveKey = `${res.result_id}/${im.key}`;
  b.addEventListener('click', () => { if (!b.dataset.saved) confirmSave(res, im, b, onSaved); });
  return b;
}

async function refreshSaved(rid, root = document) {
  try {
    for (const s of await api(`/v1/workbench/saves?result_id=${encodeURIComponent(rid)}`)) {
      const b = root.querySelector(`button[data-save-key="${rid}/${s.imageKey}"]`);
      if (b) markSaved(b, s, true);
    }
  } catch (e) { /* Java 가 꺼져 있으면 표시만 못 한다 */ }
}

function confirmSave(res, im, btn, onSaved) {
  const o = res.origin;
  const dlg = document.createElement('dialog');
  dlg.className = 'confirm';
  dlg.innerHTML = `<h2>이 결과를 DB 에 저장할까요?</h2>
    <div class="pair"><figure><img alt="원본"><figcaption><b>원본</b></figcaption></figure>
    <figure><img alt="결과"><figcaption><b>${imgLabel(im)} 결과</b></figcaption></figure></div>
    <p>아동 <b></b> · 목표 나이 <b>${im.key.slice(3)}세</b><br><span class="status">${res.result_id} / ${im.key}</span></p>
    <p class="hint">${TTL_NOTE}</p>
    <div class="row" style="margin:0"><button class="ghost" value="cancel">취소</button><button value="ok">저장</button></div>`;
  dlg.querySelector('p b').textContent = o.person_name || o.person_id;  // 이름은 DB 값이라 글자로만 넣는다
  const [src, out] = dlg.querySelectorAll('img');
  blobUrl(`/v1/workbench/persons/${o.person_id}/images/${o.source_image_id}`).then(u => src.src = u).catch(() => {});
  blobUrl(im.url).then(u => out.src = u).catch(() => {});
  dlg.addEventListener('click', async e => {
    const v = e.target.value;
    if (v === 'cancel') { dlg.close(); return; }
    if (v !== 'ok') return;
    dlg.close();
    btn.disabled = true;
    btn.textContent = '저장 중…';
    const form = new FormData();
    form.append('result_id', res.result_id);
    form.append('key', im.key);
    try {
      const saved = await api('/v1/workbench/saves', form);  // Java 가 DB 저장 성공을 돌려준 뒤에만 여기로 온다
      markSaved(btn, saved, saved.created);
      if (onSaved) onSaved(saved);
    } catch (err) {
      btn.className = 'small err';
      btn.textContent = '저장 실패 · 다시 시도';
      btn.title = err.message;
    } finally { btn.disabled = false; }
  });
  dlg.addEventListener('close', () => dlg.remove());
  document.body.append(dlg);
  dlg.showModal();
}
