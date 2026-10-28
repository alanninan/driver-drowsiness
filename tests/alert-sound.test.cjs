const {test} = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const AlertSound = require("../src/drowsiness/static/alert-sound.js");

function setup() {
    let enabled = true, clock = 0, status = "";
    const tones = [];
    const context = {
        state: "suspended", currentTime: 0, destination: {},
        async resume() { this.state = "running"; },
        createOscillator() {
            const tone = {frequency: {}, connect() {}, disconnect() {},
                start() { this.started = true; }, stop(time) { this.stoppedAt = time; }};
            tones.push(tone);
            return tone;
        },
        createGain() { return {gain: {setValueAtTime() {}, exponentialRampToValueAtTime() {}},
            connect() {}, disconnect() {}}; }
    };
    const sound = new AlertSound({enabled: () => enabled, status: s => status = s,
        createContext: () => context, now: () => clock});
    return {sound, context, tones, setEnabled: b => enabled = b, setTime: t => clock = t,
        status: () => status};
}

test("first critical alert plays immediately; subsequent alerts respect five-second cooldown", async () => {
    const s = setup();
    await s.sound.enable();
    assert.equal(s.sound.play(), true);
    assert.equal(s.tones[0].frequency.value, 740);
    assert.equal(s.tones[0].stoppedAt, .35);
    s.setTime(4999);
    assert.equal(s.sound.play(), false);
    s.setTime(5000);
    assert.equal(s.sound.play(), true);
    assert.equal(s.tones.length, 2);
});

test("mute and stop silence an active tone; reset removes the previous session cooldown", async () => {
    const s = setup();
    await s.sound.enable();
    s.sound.play();
    s.setEnabled(false);
    s.sound.stop();
    assert.equal(s.tones[0].stoppedAt, undefined);
    assert.equal(s.sound.play(), false);
    s.setEnabled(true);
    s.sound.reset();
    assert.equal(s.sound.play(), true);
});

test("suspended audio never queues stale alerts; a gesture can resume it", async () => {
    const s = setup();
    await s.sound.enable();
    s.context.state = "suspended";
    assert.equal(s.sound.play(), false);
    assert.equal(s.tones.length, 0);
    assert.match(s.status(), /paused/);
    await s.sound.enable();
    assert.equal(s.sound.play(), true);
});

test("unavailable audio and rejected resume are handled without throwing", async () => {
    const s = setup();
    s.context.resume = async () => { throw new Error("blocked"); };
    assert.equal(await s.sound.enable(), false);
    assert.match(s.status(), /unavailable/);
    s.sound.createContext = () => { throw new Error("unsupported"); };
    s.sound.context = null;
    assert.equal(await s.sound.enable(), false);
});

test("test tone does not delay the next critical alert", async () => {
    const s = setup();
    await s.sound.enable();
    assert.equal(s.sound.play(false), true);
    assert.equal(s.sound.play(), true);
});

test("mute during pending audio resume prevents playback", async () => {
    const s = setup();
    let finish;
    s.context.resume = () => new Promise(resolve => { finish = resolve; });
    const pending = s.sound.enable();
    s.setEnabled(false);
    s.context.state = "running";
    finish();
    assert.equal(await pending, false);
    assert.equal(s.sound.play(false), false);
});

test("page enables sound by default and loads audio controller before app", () => {
    const html = fs.readFileSync(path.join(__dirname, "../src/drowsiness/static/index.html"), "utf8");
    assert.match(html, /<input id="sound"[^>]*\bchecked\b/);
    assert.match(html, /id="test-sound"/);
    assert.ok(html.indexOf('/static/alert-sound.js') < html.indexOf('/static/app.js'));
});
