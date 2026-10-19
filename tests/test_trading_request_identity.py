"""Run the shipped quick-trade retry logic against a lost response."""
import json
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = r'''
const vm = require('node:vm');
let raw=''; process.stdin.setEncoding('utf8'); process.stdin.on('data',s=>raw+=s);
process.stdin.on('end',async()=>{
  const code=JSON.parse(raw), saved=new Map(), sent=[], errors=[]; let nonce=0, outcome='lost';
  const nodes={trConfirm:{value:'PLACE REAL ORDER'},trPreview:{innerHTML:''},trConfirmRow:{hidden:false}};
  const body={platform:'polymarket',slug:'fixture-market',price:.42,quantity:10,time_in_force:'FAK'};
  const context=vm.createContext({
    sessionStorage:{getItem:k=>saved.get(k)||null,setItem:(k,v)=>saved.set(k,v),removeItem:k=>saved.delete(k)},
    crypto:{randomUUID:()=>`fixture-${++nonce}`},document:{getElementById:id=>nodes[id]},
    _tradeBody:()=>({...body}),_validTradeInputs:()=>true,_trError:s=>errors.push(s),authHeaders:x=>x,
    uiIcon:()=>'',fetch:async(url,opts)=>{
      sent.push(JSON.parse(opts.body));
      if(outcome==='lost') throw new Error('fixture response lost');
      if(outcome==='invalid') return {ok:true,json:async()=>({})};
      return {ok:true,json:async()=>({submitted:true,venue_response:{acknowledgement:{acknowledged:true}}})};
    }
  });
  vm.runInContext(code.helpers+code.submit,context);
  const previewId=vm.runInContext('_trRequestId(_tradeBody())',context);
  await Promise.all([context.submitTrade(),context.submitTrade()]);
  outcome='invalid'; await context.submitTrade();
  outcome='accepted'; await context.submitTrade();
  if(nodes.trConfirm.value!=='') throw new Error('old confirmation survived a conclusive receipt');
  nodes.trConfirm.value='PLACE REAL ORDER'; await context.submitTrade();
  if(sent.length!==4 || sent[0].client_order_id!==previewId ||
     sent.slice(0,3).some(x=>x.client_order_id!==previewId) || sent[3].client_order_id===previewId)
    throw new Error('request identity changed before a conclusive receipt');
  if(!errors.some(x=>x.includes('outcome is unknown'))) throw new Error('invalid receipt was accepted');
  console.log(JSON.stringify({attempts:sent.length,same_identity_until_receipt:true,fresh_after_receipt:true}));
});
'''


@unittest.skipUnless(shutil.which('node'), 'Node unavailable for shipped browser contract')
class TradingRequestIdentityTests(unittest.TestCase):
    def test_browser_and_static_bundle_keep_identity_across_lost_and_invalid_responses(self):
        for name in ('frontend/index.html','static/index.html'):
            with self.subTest(bundle=name):
                source = (ROOT/name).read_text(encoding='utf-8')
                helpers = source[source.index('function _trRequestKey('):source.index('function _tradeBody(')]
                start = source.index('async function submitTrade(')
                end = source.index('// ── Favourite markets', start)
                result = subprocess.run([shutil.which('node'), '-e', PROBE],
                    input=json.dumps({'helpers':helpers,'submit':source[start:end]}),
                    text=True, capture_output=True, timeout=15, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(json.loads(result.stdout)['same_identity_until_receipt'])
