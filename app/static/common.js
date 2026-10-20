// 시험 페이지 공통 (aging.html, post.html): API 호출, 서버 상태, 최근 결과 목록, 결과 그리기
const $ = id => document.getElementById(id);
const radio = name => document.querySelector(`input[name=${name}]:checked`).value;
const setRadio = (name, v) => { const el = document.querySelector(`input[name=${name}][value=${v}]`); if (el) el.checked = true; };
const KIND = { aging: '나이 변환', hair: '머리', makeup: '화장', glasses: '안경' };

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
const imgLabel = im => im.age != null ? `${im.age}세` : im.key.startsWith('age') ? `${im.key.slice(3)}세` : im.key === 'face' ? '사진' : im.key;

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
