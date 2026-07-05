const API_CONFIG = {
    // Dynamic Base URL to support both unified deployments (served by Flask)
    // and separate deployments (served as a Render Static Site + Render Web Service)
    BASE_URL: (function() {
        const storedUrl = localStorage.getItem('backend_url');
        if (storedUrl) return storedUrl;

        // If running locally, check if we are on a different port than Flask (usually 5000)
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            if (window.location.port !== '5000' && window.location.port !== '') {
                return 'http://localhost:5000/api';
            }
        }
        
        // If frontend is deployed as a Render Static Site, you can set the backend URL here:
        const HARDCODED_BACKEND_URL = ''; 
        if (HARDCODED_BACKEND_URL) return HARDCODED_BACKEND_URL;

        return '/api';
    })(),
    ENDPOINTS: {
        SIGNUP:           '/auth/signup',
        LOGIN:            '/auth/login',
        CHECK_TOKEN:      '/auth/check-token',
        CHANGE_PASSWORD:  '/auth/change-password',
        FORGOT_PASSWORD:  '/auth/forgot-password',
        RESET_PASSWORD:   '/auth/reset-password',
        UPLOAD_IMAGE:     '/detection/upload-image',
        UPLOAD_VIDEO:     '/detection/upload-video',
        UPLOAD_BULK:      '/detection/upload-bulk',
        HISTORY:          '/detection/history',
        EXPORT_CSV:       '/detection/export-csv',
        USER_STATS:       '/detection/stats',
        DETECTION_DETAIL: '/detection/detection',
        ADMIN_USERS:      '/admin/users',
        ADMIN_DETECTIONS: '/admin/detections',
        ADMIN_STATS:      '/admin/dashboard-stats',
        ADMIN_DASHBOARD:  '/admin/dashboard-stats',
        ADMIN_USER_DETAIL:'/admin/user',
    }
};

const API = {
    async request(endpoint, options = {}) {
        const url     = `${API_CONFIG.BASE_URL}${endpoint}`;
        const token   = Storage.getToken();
        const headers = { 'Content-Type': 'application/json', ...options.headers };
        if (token && !options.skipAuth) headers['Authorization'] = `Bearer ${token}`;
        const response = await fetch(url, { ...options, headers });
        const data     = await response.json();
        if (!response.ok) throw new Error(data.error || 'Request failed');
        return data;
    },
    uploadFile(endpoint, file, onProgress) {
        return new Promise((resolve, reject) => {
            const url  = `${API_CONFIG.BASE_URL}${endpoint}`;
            const xhr  = new XMLHttpRequest();
            const form = new FormData();
            form.append('file', file);
            if (onProgress) {
                xhr.upload.addEventListener('progress', e => {
                    if (e.lengthComputable) onProgress((e.loaded/e.total)*100);
                });
            }
            xhr.addEventListener('load', () => {
                try {
                    const data = JSON.parse(xhr.responseText);
                    if (xhr.status >= 200 && xhr.status < 300) resolve(data);
                    else reject(new Error(data.error || 'Upload failed'));
                } catch { reject(new Error('Invalid server response')); }
            });
            xhr.addEventListener('error', () => reject(new Error('Network error')));
            xhr.open('POST', url);
            xhr.setRequestHeader('Authorization', `Bearer ${Storage.getToken()}`);
            xhr.send(form);
        });
    },
    uploadBulk(endpoint, files, onProgress) {
        return new Promise((resolve, reject) => {
            const url  = `${API_CONFIG.BASE_URL}${endpoint}`;
            const xhr  = new XMLHttpRequest();
            const form = new FormData();
            files.forEach(f => form.append('files', f));
            if (onProgress) {
                xhr.upload.addEventListener('progress', e => {
                    if (e.lengthComputable) onProgress((e.loaded/e.total)*100);
                });
            }
            xhr.addEventListener('load', () => {
                try {
                    const data = JSON.parse(xhr.responseText);
                    if (xhr.status >= 200 && xhr.status < 300) resolve(data);
                    else reject(new Error(data.error || 'Upload failed'));
                } catch { reject(new Error('Invalid server response')); }
            });
            xhr.addEventListener('error', () => reject(new Error('Network error')));
            xhr.open('POST', url);
            xhr.setRequestHeader('Authorization', `Bearer ${Storage.getToken()}`);
            xhr.send(form);
        });
    },
    async delete(endpoint) { return this.request(endpoint, { method: 'DELETE' }); }
};

const Storage = {
    setToken(t)       { localStorage.setItem('token', t); },
    getToken()        { return localStorage.getItem('token'); },
    removeToken()     { localStorage.removeItem('token'); },
    setUser(u)        { localStorage.setItem('user', JSON.stringify(u)); },
    getUser()         { const u=localStorage.getItem('user'); return u?JSON.parse(u):null; },
    removeUser()      { localStorage.removeItem('user'); },
    clear()           { localStorage.clear(); },
    isAuthenticated() { return !!this.getToken(); },
    isAdmin()         { const u=this.getUser(); return u&&u.is_admin; },
};

const Redirect = {
    toLogin()          { window.location.href='/login.html'; },
    toDashboard()      { window.location.href='/dashboard.html'; },
    toAdminDashboard() { window.location.href='/admin-dashboard.html'; },
};

function requireAuth()  { if(!Storage.isAuthenticated()){Redirect.toLogin();return false;} return true; }
function requireAdmin() { if(!Storage.isAuthenticated()||!Storage.isAdmin()){Redirect.toLogin();return false;} return true; }
function logout()       { Storage.clear(); Redirect.toLogin(); }

// ── Toast notifications ────────────────────────────────────────────────────
const Toast = {
    _container: null,
    _init() {
        if (this._container) return;
        this._container = document.createElement('div');
        this._container.id = 'toast-container';
        this._container.style.cssText =
            'position:fixed;bottom:1.5rem;right:1.5rem;z-index:9999;display:flex;flex-direction:column;gap:.5rem;';
        document.body.appendChild(this._container);
    },
    show(message, type='info', duration=3500) {
        this._init();
        const colours = {
            success: { bg:'#d1fae5', border:'#059669', text:'#065f46', icon:'✅' },
            error:   { bg:'#fee2e2', border:'#dc2626', text:'#991b1b', icon:'❌' },
            info:    { bg:'#e0f2fe', border:'#0284c7', text:'#075985', icon:'ℹ️'  },
            warning: { bg:'#fff7ed', border:'#f59e0b', text:'#92400e', icon:'⚠️'  },
        };
        const c = colours[type] || colours.info;
        const el = document.createElement('div');
        el.style.cssText = `background:${c.bg};border-left:4px solid ${c.border};
            color:${c.text};padding:.85rem 1.25rem;border-radius:8px;
            box-shadow:0 4px 12px rgba(0,0,0,.15);max-width:320px;
            font-size:.92rem;display:flex;align-items:center;gap:.6rem;
            animation:toastIn .25s ease-out;`;
        el.innerHTML = `<span>${c.icon}</span><span>${message}</span>`;
        this._container.appendChild(el);
        setTimeout(() => {
            el.style.opacity='0'; el.style.transform='translateX(100%)';
            el.style.transition='all .3s'; setTimeout(()=>el.remove(), 300);
        }, duration);
    },
    success(m, d) { this.show(m,'success',d); },
    error(m,   d) { this.show(m,'error',  d); },
    info(m,    d) { this.show(m,'info',   d); },
    warning(m, d) { this.show(m,'warning',d); },
};

// ── Dark mode ──────────────────────────────────────────────────────────────
const DarkMode = {
    _key: 'darkMode',
    isEnabled() { return localStorage.getItem(this._key) === 'true'; },
    enable()  {
        document.documentElement.setAttribute('data-theme','dark');
        localStorage.setItem(this._key,'true');
        this._updateToggle(true);
    },
    disable() {
        document.documentElement.removeAttribute('data-theme');
        localStorage.setItem(this._key,'false');
        this._updateToggle(false);
    },
    toggle() { this.isEnabled() ? this.disable() : this.enable(); },
    init()   { if (this.isEnabled()) this.enable(); else this._updateToggle(false); },
    _updateToggle(on) {
        const btn = document.getElementById('dark-toggle');
        if (btn) btn.textContent = on ? '☀️' : '🌙';
    },
};

document.addEventListener('DOMContentLoaded', () => {
    DarkMode.init();
    // Inject toast keyframe once
    if (!document.getElementById('toast-style')) {
        const s = document.createElement('style');
        s.id = 'toast-style';
        s.textContent = '@keyframes toastIn{from{opacity:0;transform:translateX(30px)}to{opacity:1;transform:translateX(0)}}';
        document.head.appendChild(s);
    }
});
