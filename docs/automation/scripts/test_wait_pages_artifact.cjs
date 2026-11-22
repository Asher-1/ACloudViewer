'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const {waitPagesArtifact} = require('./wait_pages_artifact.cjs')

const name = 'github-pages-45032677079-1'
const expected = {name, id: 14763475255, size: 668022996}
const old = ['docs-output', 'website-deploy', 'build-archive'].map((name, i) =>
  ({name, id: i + 1, size: 100}))

function setup(inventories, options = {}) {
  let clock = 0
  let calls = 0
  const waits = []
  const client = {listArtifacts: async () => {
    const value = inventories[Math.min(calls++, inventories.length - 1)]
    if (value instanceof Error) throw value
    return value
  }}
  return {
    run: () => waitPagesArtifact({client, name, expectedId: String(expected.id),
      timeoutMs: 12000, now: () => clock,
      sleep: async ms => {waits.push(ms); clock += ms}, log: () => {}, ...options}),
    calls: () => calls, waits
  }
}

test('stale three-artifact list then precise upload receipt becomes visible', async () => {
  const gate = setup([{artifacts: old}, {artifacts: [...old, expected]}])
  assert.deepEqual(await gate.run(), expected)
  assert.equal(gate.calls(), 2)
  assert.deepEqual(gate.waits, [1000])
})

test('visible exact artifact returns immediately without sleeping', async () => {
  const gate = setup([{artifacts: [expected]}])
  assert.deepEqual(await gate.run(), expected)
  assert.deepEqual(gate.waits, [])
})

test('missing artifact exhausts a bounded deadline', async () => {
  const gate = setup([{artifacts: old}])
  await assert.rejects(gate.run(), /visibility deadline/)
  assert.deepEqual(gate.waits, [1000, 2000, 4000, 5000])
  assert.equal(gate.calls(), 4)
})

test('name collision or duplicate name fails immediately, never deploys another ID', async () => {
  for (const artifacts of [[{...expected, id: expected.id + 1}],
    [expected, {...expected, id: expected.id + 1}], [{...expected, name: 'other'}]]) {
    const gate = setup([{artifacts}])
    await assert.rejects(gate.run(), /match|Multiple/)
    assert.equal(gate.calls(), 1)
    assert.deepEqual(gate.waits, [])
  }
})

test('empty artifact and malformed service inventories are not absence', async () => {
  for (const response of [{artifacts: [{...expected, size: 0}]}, {},
    {artifacts: null}, {artifacts: [{...expected, id: NaN}]},
    {artifacts: [{...expected, size: '668022996'}]}]) {
    const gate = setup([response])
    await assert.rejects(gate.run(), /content|inventory/)
    assert.equal(gate.calls(), 1)
    assert.deepEqual(gate.waits, [])
  }
})

test('transport or permission error is not retried as not-yet-visible', async () => {
  const error = new Error('permission failure')
  const gate = setup([error])
  await assert.rejects(gate.run(), value => value === error)
  assert.equal(gate.calls(), 1)
  assert.deepEqual(gate.waits, [])
})

test('invalid upload ID is rejected before a service call', async () => {
  for (const expectedId of ['', undefined, 'NaN', '1.5', '0', '-1', '1e3', '9007199254740993']) {
    const gate = setup([{artifacts: [expected]}], {expectedId})
    await assert.rejects(gate.run(), /Invalid expected/)
    assert.equal(gate.calls(), 0)
  }
})

test('hung runtime lookup is also bounded by the gate deadline', async () => {
  await assert.rejects(waitPagesArtifact({name, expectedId: expected.id,
    timeoutMs: 10, client: {listArtifacts: () => new Promise(() => {})}, log: () => {}}),
  /visibility deadline/)
})
