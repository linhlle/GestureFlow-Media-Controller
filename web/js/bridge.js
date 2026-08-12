// Optional client for the local GestureFlow bridge.
//
// The bridge is a convenience, not a requirement: the site's primary handoff is
// downloading a config file. When the desktop app is running `gestureflow
// bridge`, it serves this same UI from http://127.0.0.1:8765, so the page
// origin is loopback and the WebSocket is same-origin -- no mixed-content rule
// applies in any browser.
//
// When the page is served from https:// instead, connecting to ws://localhost
// is attempted once and allowed to fail quietly. Browsers disagree about
// whether to permit it, and Safari -- the default browser on the macOS this
// app targets -- is the strictest. A failure here is expected and costs the
// user nothing.

const DEFAULT_PORT = 8765;
const RECONNECT_MS = 4000;

export class BridgeClient {
  constructor(port = DEFAULT_PORT) {
    this.port = port;
    this.socket = null;
    this.connected = false;
    this.onStatus = () => {};
    this.onState = () => {};
    this._timer = null;
    this._everConnected = false;
  }

  get url() {
    // Same-origin when served by the bridge itself; explicit loopback otherwise.
    if (window.location.port === String(this.port)
        && (window.location.hostname === '127.0.0.1'
            || window.location.hostname === 'localhost')) {
      const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      return `${scheme}//${window.location.host}/ws`;
    }
    return `ws://127.0.0.1:${this.port}/ws`;
  }

  tryConnect() {
    if (this.socket) return;
    let socket;
    try {
      socket = new WebSocket(this.url);
    } catch {
      this._scheduleRetry();
      return;
    }
    this.socket = socket;

    socket.addEventListener('open', () => {
      this.connected = true;
      this._everConnected = true;
      this.onStatus(true);
    });

    socket.addEventListener('message', (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === 'state') this.onState(payload.state);
      } catch {
        // A malformed frame is not worth tearing the connection down for.
      }
    });

    socket.addEventListener('close', () => {
      this.socket = null;
      if (this.connected) {
        this.connected = false;
        this.onStatus(false);
      }
      this._scheduleRetry();
    });

    socket.addEventListener('error', () => {
      // 'close' always follows; handle the state change there so the two paths
      // do not both fire onStatus.
      socket.close();
    });
  }

  _scheduleRetry() {
    if (this._timer) return;
    // Only keep retrying if the bridge was there at some point. Otherwise one
    // attempt is enough -- most visitors will never run it, and a console full
    // of failed connections is noise.
    if (!this._everConnected) return;
    this._timer = setTimeout(() => {
      this._timer = null;
      this.tryConnect();
    }, RECONNECT_MS);
  }

  async pushConfig(config) {
    if (!this.connected || !this.socket) {
      throw new Error('the local bridge is not connected');
    }
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.socket.removeEventListener('message', handler);
        reject(new Error('the app did not respond'));
      }, 5000);

      const handler = (event) => {
        let payload;
        try {
          payload = JSON.parse(event.data);
        } catch {
          return;
        }
        if (payload.type !== 'config_result') return;
        clearTimeout(timeout);
        this.socket.removeEventListener('message', handler);
        if (payload.ok) resolve(payload);
        else reject(new Error(payload.error || 'the app rejected the config'));
      };

      this.socket.addEventListener('message', handler);
      this.socket.send(JSON.stringify({ type: 'set_config', config }));
    });
  }

  close() {
    if (this._timer) clearTimeout(this._timer);
    this._timer = null;
    if (this.socket) this.socket.close();
    this.socket = null;
  }
}
