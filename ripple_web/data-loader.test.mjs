import test from 'node:test';
import assert from 'node:assert/strict';
import {DATASETS,loadDatasets} from './data-loader.mjs';
test('one optional HTTP failure preserves all available sections', async () => {
  const requests=[];
  const result=await loadDatasets(async (path) => {
    requests.push(path);
    return path.includes('acceptance') ? {ok:false,status:503} : {ok:true,json:async()=>({path})};
  });
  assert.equal(Object.keys(result.data).length,7);
  assert.equal(result.errors.acceptance,'HTTP 503');
  assert.ok(result.data.replays && result.data.evaluation);
  assert.deepEqual(requests,DATASETS.map((name)=>`data/${name}.json`));
});
test('JSON decode failure does not discard other data', async () => {
  const result=await loadDatasets(async (path)=>({ok:true,json:async()=>{
    if(path.includes('graph'))throw new Error('invalid JSON');
    return {};
  }}));
  assert.equal(result.errors.graph,'invalid JSON');
  assert.ok(result.data.radar);
});
test('network failure returns explicit errors with no fabricated data', async () => {
  const result=await loadDatasets(async()=>{throw new Error('offline')});
  assert.deepEqual(result.data,{});
  assert.equal(Object.keys(result.errors).length,8);
});
