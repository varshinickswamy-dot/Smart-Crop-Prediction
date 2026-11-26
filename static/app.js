const $ = (id) => document.getElementById(id);
async function request(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || "Request failed. Please try again.");
  return data;
}
function show(id, message, error=false) {
  const el=$(id); el.textContent=message; el.classList.toggle("error",error);
}
function setBusy(form,busy) {
  const button=form.querySelector("button"); button.disabled=busy;
  if(busy){button.dataset.label=button.textContent;button.textContent="Please wait…";}
  else{button.textContent=button.dataset.label||button.textContent;}
}
async function submit(formId, resultId, handler) {
  const form=$(formId);
  form.addEventListener("submit",async(event)=>{
    event.preventDefault();setBusy(form,true);show(resultId,"");
    try{show(resultId,await handler(new FormData(form),form));}
    catch(error){show(resultId,error.message,true);}
    finally{setBusy(form,false);}
  });
}
submit("crop-form","crop-result",async(data)=>{
  const payload=Object.fromEntries(data.entries());
  const result=await request("/predict_crop",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
  return "Recommended crop: "+result.crop;
});
submit("disease-form","disease-result",async(data)=>{
  const result=await request("/predict_disease",{method:"POST",body:data});
  return result.disease+" · Confidence: "+result.confidence+"%";
});
submit("yield-form","yield-result",async(data)=>{
  const result=await request("/predict_yield",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(Object.fromEntries(data.entries()))});
  return "Estimated yield: "+result.yield;
});
(async()=>{
  try{
    const health=await request("/health");
    const ready=Object.values(health.features).filter(Boolean).length;
    $("service-status").textContent=ready+" of 4 services ready";
  }catch{$("service-status").textContent="Service status unavailable";}
  try{
    const options=await request("/get-yield-options");
    $("area-options").innerHTML=options.areas.map(v=>"<option></option>").join("");
    options.areas.forEach(v=>{const o=document.createElement("option");o.value=v;$("area-options").appendChild(o);});
    options.crops.forEach(v=>{const o=document.createElement("option");o.value=v;$("crop-options").appendChild(o);});
    $("yield-hint").textContent="Select an area and crop from the available dataset options.";
  }catch(error){$("yield-hint").textContent=error.message;}
})();
