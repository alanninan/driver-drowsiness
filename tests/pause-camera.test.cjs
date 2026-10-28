const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function setup() {
    const elements = new Map();
    const el = id => {
        if (!elements.has(id)) elements.set(id, {
            checked: true, textContent: '', style: {}, value: 1,
            videoWidth: 640, videoHeight: 480, pause() {},
            replaceChildren() {}, append() {}, setAttribute() {},
            querySelector() { return {setAttribute() {}}; },
            getBoundingClientRect() { return {width: 600}; }
        });
        return elements.get(id);
    };
    let blobCallback;
    const canvas = {getContext: () => ({drawImage() {}}), toBlob: fn => blobCallback = fn};
    const sent = [], track = {enabled: true};
    const sandbox = {document: {getElementById: el, createElement: tag => tag === 'canvas' ? canvas : {},
        createTextNode: s => s, body: {dataset: {}}},
        AlertSound: class {stop() {} reset() {} enable() {}},
        window: {}, WebSocket: {OPEN: 1}, performance: {now: () => 10000},
        clearTimeout() {}, sent, track, setTimeout, Uint8Array, DataView};
    vm.createContext(sandbox);
    const source = fs.readFileSync(path.join(__dirname, '../src/drowsiness/static/app.js'), 'utf8');
    vm.runInContext(source.split('window.addEventListener("resize", drawChart);')[0], sandbox);
    vm.runInContext(`running = true; mode = 'live'; calibrated = true;
        socket = {readyState: 1, send: data => sent.push(data)};
        stream = {getVideoTracks: () => [track]};`, sandbox);
    return {sandbox, sent, track, el, blob: () => blobCallback};
}

test('pause disables camera capture, silences readings, and sends pause command', () => {
    const s = setup();
    vm.runInContext('togglePause(); sendFrame(generation);', s.sandbox);
    assert.equal(s.track.enabled, false);
    assert.equal(s.sent.length, 1);
    assert.equal(JSON.parse(s.sent[0]).action, 'pause');
    assert.equal(s.el('pause').disabled, true); // Wait for server acknowledgement.
    assert.equal(s.el('video').hidden, true);
    assert.equal(s.el('state').textContent, 'Camera paused');
    assert.equal(s.blob(), undefined);
});

test('asynchronous frame captured before pause cannot leak into resumed session', async () => {
    const s = setup();
    vm.runInContext('sendFrame(generation); togglePause(); paused = false; pausePending = false;', s.sandbox);
    await s.blob()({arrayBuffer: async () => new ArrayBuffer(4)});
    assert.equal(s.sent.length, 1); // Only the pause command, no stale image.
});

test('pause is unavailable before calibration and for simulated sessions', () => {
    const s = setup();
    vm.runInContext('calibrated = false; togglePause(); calibrated = true; mode = "demo"; togglePause();', s.sandbox);
    assert.equal(s.sent.length, 0);
    assert.equal(s.track.enabled, true);
});
