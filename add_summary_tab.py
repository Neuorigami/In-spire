import sys

file_path = r'index.html'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add nav item
old_nav = '<div class="nav-item" onclick="switchTab(\'predictor\')" id="nav-predictor">Predictor</div>'
new_nav = old_nav + '\n      <div class="nav-item" onclick="switchTab(\'summary\')" id="nav-summary">📊 Data Summary</div>'
content = content.replace(old_nav, new_nav)

# 2. Add summary section HTML after predictor-section closing div
# Find the closing of predictor section (before the script tag)
old_script_start = '    <script>'
summary_html = '''
    <!-- ── Data Summary Section ─────────────────────────────────────────── -->
    <div id="summary-section" style="display:none;">
      <div class="home-hero" style="margin-bottom:30px;">
        <h1 style="font-size:36px;">📊 Data Summary</h1>
        <p style="color:var(--muted);">Overview of all patient data used to train the model</p>
      </div>

      <!-- Stats Cards -->
      <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px; margin-bottom:30px;">
        <div class="home-card" style="text-align:center; padding:20px;">
          <div style="font-size:40px; font-weight:800; color:var(--accent);" id="stat-total">—</div>
          <div style="color:var(--muted); margin-top:6px;">Total Patients</div>
        </div>
        <div class="home-card" style="text-align:center; padding:20px;">
          <div style="font-size:40px; font-weight:800; color:#4ade80;" id="stat-positive">—</div>
          <div style="color:var(--muted); margin-top:6px;">Positive Outcomes</div>
        </div>
        <div class="home-card" style="text-align:center; padding:20px;">
          <div style="font-size:40px; font-weight:800; color:var(--danger);" id="stat-negative">—</div>
          <div style="color:var(--muted); margin-top:6px;">Negative Outcomes</div>
        </div>
        <div class="home-card" style="text-align:center; padding:20px;">
          <div style="font-size:40px; font-weight:800; color:var(--warn);" id="stat-avg-age">—</div>
          <div style="color:var(--muted); margin-top:6px;">Avg. Patient Age</div>
        </div>
        <div class="home-card" style="text-align:center; padding:20px;">
          <div style="font-size:40px; font-weight:800; color:var(--accent2);" id="stat-mongo">—</div>
          <div style="color:var(--muted); margin-top:6px;">New Patients (MongoDB)</div>
        </div>
      </div>

      <!-- Download Button -->
      <div style="display:flex; gap:12px; margin-bottom:30px; flex-wrap:wrap;">
        <button class="btn-predict" onclick="downloadAllData()" style="flex:1; min-width:200px;">
          ⬇ Download All Training Data (Excel)
        </button>
        <button class="btn-predict" onclick="loadSummary()" style="flex:1; min-width:200px; background:linear-gradient(135deg,#334155,#475569);">
          🔄 Refresh Data
        </button>
      </div>

      <!-- Condition Breakdown -->
      <div class="home-card" style="margin-bottom:20px;">
        <h3>Condition Breakdown</h3>
        <div id="condition-breakdown" style="margin-top:10px;">Loading...</div>
      </div>

      <!-- Gender Breakdown -->
      <div class="home-card" style="margin-bottom:20px;">
        <h3>Gender Breakdown</h3>
        <div id="gender-breakdown" style="margin-top:10px;">Loading...</div>
      </div>

      <!-- Recent Patients Table -->
      <div class="home-card">
        <h3>Recently Added Patients (MongoDB)</h3>
        <div id="recent-patients" style="margin-top:10px; overflow-x:auto;">Loading...</div>
      </div>
    </div>

'''
content = content.replace(old_script_start, summary_html + '    <script>')

# 3. Update switchTab function
old_switchtab = '''    function switchTab(tab) {
      document.getElementById('nav-home').classList.remove('active');
      document.getElementById('nav-predictor').classList.remove('active');
      document.getElementById('nav-add').classList.remove('active');
      document.getElementById('home-section').style.display = 'none';
      document.getElementById('predictor-section').style.display = 'none';'''

new_switchtab = '''    function switchTab(tab) {
      document.getElementById('nav-home').classList.remove('active');
      document.getElementById('nav-predictor').classList.remove('active');
      document.getElementById('nav-add').classList.remove('active');
      document.getElementById('nav-summary').classList.remove('active');
      document.getElementById('home-section').style.display = 'none';
      document.getElementById('predictor-section').style.display = 'none';
      document.getElementById('summary-section').style.display = 'none';'''

content = content.replace(old_switchtab, new_switchtab)

# 4. Add summary tab case in switchTab
old_add_case = '''        document.getElementById('spo2-section-label').textContent = 'SpO₂ Readings Post-Intubation';
      }
    }'''

new_add_case = '''        document.getElementById('spo2-section-label').textContent = 'SpO₂ Readings Post-Intubation';
      } else if (tab === 'summary') {
        document.getElementById('nav-summary').classList.add('active');
        document.getElementById('summary-section').style.display = 'block';
        loadSummary();
      }
    }'''

content = content.replace(old_add_case, new_add_case)

# 5. Add JS functions before "// Enter key shortcut"
summary_js = '''
    // ── Data Summary Functions ────────────────────────────────────────────────
    async function loadSummary() {
      try {
        // Load MongoDB patients
        const resp = await fetch(`${API}/patients/all`);
        const json = await resp.json();
        const mongoPatients = json.patients || [];

        // Stats
        document.getElementById('stat-mongo').textContent = mongoPatients.length;

        const totalPositive = mongoPatients.filter(p => p.outcome === 1).length;
        const totalNegative = mongoPatients.filter(p => p.outcome === 0).length;
        const avgAge = mongoPatients.length > 0
          ? Math.round(mongoPatients.reduce((s,p) => s + p.age, 0) / mongoPatients.length)
          : '—';

        document.getElementById('stat-positive').textContent = totalPositive;
        document.getElementById('stat-negative').textContent = totalNegative;
        document.getElementById('stat-avg-age').textContent = avgAge;
        document.getElementById('stat-total').textContent = mongoPatients.length + ' (DB)';

        // Condition breakdown
        const condMap = {};
        mongoPatients.forEach(p => { condMap[p.condition] = (condMap[p.condition]||0)+1; });
        const condHtml = Object.entries(condMap).sort((a,b)=>b[1]-a[1]).map(([c,n]) => `
          <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
            <div style="width:160px; color:var(--text); font-size:14px;">${c}</div>
            <div style="flex:1; background:rgba(255,255,255,0.05); border-radius:4px; height:20px; overflow:hidden;">
              <div style="width:${Math.round(n/mongoPatients.length*100)}%; background:linear-gradient(90deg,var(--accent),var(--accent2)); height:100%; border-radius:4px;"></div>
            </div>
            <div style="width:30px; color:var(--accent); font-weight:600;">${n}</div>
          </div>`).join('');
        document.getElementById('condition-breakdown').innerHTML = condHtml || '<p style="color:var(--muted)">No data yet</p>';

        // Gender breakdown
        const genMap = {};
        mongoPatients.forEach(p => { genMap[p.gender] = (genMap[p.gender]||0)+1; });
        const genHtml = Object.entries(genMap).map(([g,n]) => `
          <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
            <div style="width:80px; color:var(--text); font-size:14px;">${g}</div>
            <div style="flex:1; background:rgba(255,255,255,0.05); border-radius:4px; height:20px; overflow:hidden;">
              <div style="width:${Math.round(n/mongoPatients.length*100)}%; background:linear-gradient(90deg,#a855f7,#ec4899); height:100%; border-radius:4px;"></div>
            </div>
            <div style="width:30px; color:#a855f7; font-weight:600;">${n}</div>
          </div>`).join('');
        document.getElementById('gender-breakdown').innerHTML = genHtml || '<p style="color:var(--muted)">No data yet</p>';

        // Recent patients table
        const recent = mongoPatients.slice(-10).reverse();
        if (recent.length === 0) {
          document.getElementById('recent-patients').innerHTML = '<p style="color:var(--muted)">No patients added yet via Add Patient tab.</p>';
        } else {
          const tableHtml = `
            <table class="home-table">
              <thead><tr>
                <th>Age</th><th>Gender</th><th>Condition</th><th>BPM</th>
                <th>SpO₂ Before</th><th>SpO₂ 30min</th><th>Outcome</th><th>Added At</th>
              </tr></thead>
              <tbody>
                ${recent.map(p => `<tr>
                  <td>${p.age}</td>
                  <td>${p.gender}</td>
                  <td>${p.condition}</td>
                  <td>${p.bpm}</td>
                  <td>${p.spo2_before}%</td>
                  <td>${p.spo2_30min}%</td>
                  <td style="color:${p.outcome===1?'#4ade80':'var(--danger)'}; font-weight:600;">
                    ${p.outcome===1?'✓ Positive':'✗ Negative'}
                  </td>
                  <td style="color:var(--muted); font-size:12px;">${p.added_at ? p.added_at.split('T')[0] : '—'}</td>
                </tr>`).join('')}
              </tbody>
            </table>`;
          document.getElementById('recent-patients').innerHTML = tableHtml;
        }

      } catch(e) {
        document.getElementById('stat-total').textContent = 'Error';
        document.getElementById('condition-breakdown').innerHTML = '<p style="color:var(--danger)">Cannot reach API. Make sure backend is online.</p>';
      }
    }

    async function downloadAllData() {
      try {
        const resp = await fetch(`${API}/patients/all`);
        if (!resp.ok) throw new Error("API not reachable");
        const json = await resp.json();
        if (!json.patients || json.patients.length === 0) {
          alert("No patient data in MongoDB yet!");
          return;
        }
        const headers = ['age','gender','condition','comorbidity','bpm','mode','peep','lpm',
          'pulse','bp_systolic','bp_diastolic','gcs_score','cvs_score',
          'spo2_before','spo2_5min','spo2_10min','spo2_15min',
          'spo2_20min','spo2_25min','spo2_30min','outcome','added_at'];
        let csv = headers.join(',') + '\\n';
        json.patients.forEach(p => {
          csv += headers.map(h => `"${p[h] ?? ''}"`).join(',') + '\\n';
        });
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `AmbuPredict_PatientData_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        URL.revokeObjectURL(url);
      } catch(e) {
        alert('Download failed: ' + e.message);
      }
    }

'''

content = content.replace('// Enter key shortcut', summary_js + '    // Enter key shortcut')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Summary tab added successfully!")
