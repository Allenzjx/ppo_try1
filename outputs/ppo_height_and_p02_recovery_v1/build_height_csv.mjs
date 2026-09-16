// Flat scientific candidate measurements. No extra Excel deliverable.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {Workbook} from '@oai/artifact-tool';
const [input,output,verificationMode] = process.argv.slice(2);
assert(verificationMode === undefined || verificationMode === '--verify-existing');
assert(input && output, 'source rows JSON and new CSV output required');
const payload=JSON.parse(await fs.readFile(input,'utf8'));
assert(Array.isArray(payload.rows) && payload.rows.length);
const keys=[...new Set(payload.rows.flatMap(Object.keys))];
const rows=payload.rows.map(r=>keys.map(k=>r[k]??null));
assert(rows.every(r=>r.every(v=>v===null || ['string','boolean'].includes(typeof v) || typeof v==='number'&&Number.isFinite(v))),
       'one scalar variable per column; missing is null, never manufactured zero');
const wb=Workbook.create(), sheet=wb.worksheets.add('Candidates');
sheet.getRangeByIndexes(0,0,rows.length+1,keys.length).values=[keys,...rows];
sheet.showGridLines=false;
sheet.getRangeByIndexes(0,0,rows.length+1,keys.length).format={font:{name:'Arial',size:10},columnWidth:36,rowHeight:54,wrapText:true};
sheet.getRangeByIndexes(0,0,1,keys.length).format={font:{name:'Arial',size:10,bold:true},fill:'#E8EEF4',wrapText:true,rowHeight:48};
wb.recalculate();
const inspect=await wb.inspect({kind:'region',sheetId:sheet.name,range:'A1:H8',maxChars:2000,tableMaxRows:7,tableMaxCols:8});
const preview=await wb.render({sheetName:sheet.name,range:`A1:H${Math.min(rows.length+1,8)}`,scale:1,format:'png'});
await fs.writeFile(path.join(path.dirname(output),'height_candidates_preview.png'),new Uint8Array(await preview.arrayBuffer()));
const saved=sheet.getRangeByIndexes(0,0,rows.length+1,keys.length).values.map(r=>r.map(v=>v??null));
assert.deepEqual(saved,[keys,...rows]);
const quote=v=>v===null?'':typeof v==='string'?'"'+v.replaceAll('"','""')+'"':String(v);
const csv=saved.map(r=>r.map(quote).join(',')).join('\r\n')+'\r\n';
if (verificationMode === '--verify-existing') assert.equal(await fs.readFile(output,'utf8'),csv);
else await fs.writeFile(output,csv,{flag:'wx'});
const loaded=await Workbook.fromCSV(await fs.readFile(output,'utf8'),{sheetName:'Reloaded'});
assert.deepEqual(loaded.worksheets.getItemAt(0).getRangeByIndexes(0,0,rows.length+1,keys.length).values.map(r=>r.map(v=>v==null?'':String(v))),
                 saved.map(r=>r.map(v=>v==null?'':String(v))));
await fs.writeFile(path.join(path.dirname(output),'height_csv_receipt.json'),JSON.stringify({input,output,rows:rows.length,
  columns:keys.length,typed_cells_verified:true,csv_roundtrip_verified:true,extra_xlsx_created:false,inspect},null,2));
console.log(JSON.stringify({output,rows:rows.length,columns:keys.length}));
