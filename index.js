!function(e){function t(t){for(var n,o,i=t[0],a=t[1],c=0,u=[];c<i.length;c++)o=i[c],Object.prototype.hasOwnProperty.call(r,o)&&r[o]&&u.push(r[o][0]),r[o]=0;for(n in a)Object.prototype.hasOwnProperty.call(a,n)&&(e[n]=a[n]);for(l&&l(t);u.length;)u.shift()()}var n={},r={0:0};var o={};var i={3:function(){return{"./index_bg.js":{__wbg_new_59cb74e423758ede:function(){return n[2].exports.b()},__wbg_stack_558ba5917b466edd:function(e,t){return n[2].exports.c(e,t)},__wbg_error_4bb6c2a97407129a:function(e,t){return n[2].exports.a(e,t)},__wbindgen_object_drop_ref:function(e){return n[2].exports.d(e)}}}}};function a(t){if(n[t])return n[t].exports;var r=n[t]={i:t,l:!1,exports:{}};return e[t].call(r.exports,r,r.exports,a),r.l=!0,r.exports}a.e=function(e){var t=[],n=r[e];if(0!==n)if(n)t.push(n[2]);else{var c=new Promise((function(t,o){n=r[e]=[t,o]}));t.push(n[2]=c);var u,s=document.createElement("script");s.charset="utf-8",s.timeout=120,a.nc&&s.setAttribute("nonce",a.nc),s.src=function(e){return a.p+""+({}[e]||e)+".js"}(e);var l=new Error;u=function(t){s.onerror=s.onload=null,clearTimeout(f);var n=r[e];if(0!==n){if(n){var o=t&&("load"===t.type?"missing":t.type),i=t&&t.target&&t.target.src;l.message="Loading chunk "+e+" failed.\n("+o+": "+i+")",l.name="ChunkLoadError",l.type=o,l.request=i,n[1](l)}r[e]=void 0}};var f=setTimeout((function(){u({type:"timeout",target:s})}),12e4);s.onerror=s.onload=u,document.head.appendChild(s)}return({1:[3]}[e]||[]).forEach((function(e){var n=o[e];if(n)t.push(n);else{var r,c=i[e](),u=fetch(a.p+""+{3:"f415a8b1521c469791a0"}[e]+".module.wasm");if(c instanceof Promise&&"function"==typeof WebAssembly.compileStreaming)r=Promise.all([WebAssembly.compileStreaming(u),c]).then((function(e){return WebAssembly.instantiate(e[0],e[1])}));else if("function"==typeof WebAssembly.instantiateStreaming)r=WebAssembly.instantiateStreaming(u,c);else{r=u.then((function(e){return e.arrayBuffer()})).then((function(e){return WebAssembly.instantiate(e,c)}))}t.push(o[e]=r.then((function(t){return a.w[e]=(t.instance||t).exports})))}})),Promise.all(t)},a.m=e,a.c=n,a.d=function(e,t,n){a.o(e,t)||Object.defineProperty(e,t,{enumerable:!0,get:n})},a.r=function(e){"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},a.t=function(e,t){if(1&t&&(e=a(e)),8&t)return e;if(4&t&&"object"==typeof e&&e&&e.__esModule)return e;var n=Object.create(null);if(a.r(n),Object.defineProperty(n,"default",{enumerable:!0,value:e}),2&t&&"string"!=typeof e)for(var r in e)a.d(n,r,function(t){return e[t]}.bind(null,r));return n},a.n=function(e){var t=e&&e.__esModule?function(){return e.default}:function(){return e};return a.d(t,"a",t),t},a.o=function(e,t){return Object.prototype.hasOwnProperty.call(e,t)},a.p="",a.oe=function(e){throw console.error(e),e},a.w={};var c=window.webpackJsonp=window.webpackJsonp||[],u=c.push.bind(c);c.push=t,c=c.slice();for(var s=0;s<c.length;s++)t(c[s]);var l=u;a(a.s=0)}([function(e,t,n){(async()=>{const{scan:e}=await n.e(1).then(n.bind(null,1)).catch(console.error),t=document.getElementById("video"),r=document.getElementById("canvas"),o=document.getElementById("result"),i=document.getElementById("rescan"),a=r.getContext("2d"),c={video:{facingMode:{exact:"environment"},width:{exact:960},height:{exact:720}},audio:!1};try{const n=await navigator.mediaDevices.getUserMedia(c);function u(){if(t.readyState>=2){const n=t.videoWidth,i=t.videoHeight;r.width=n,r.height=i,a.drawImage(t,0,0),a.strokeRect((n-360)/2,(i-360)/2,360,360);try{const t=e(360,360,a.getImageData((n-360)/2,(i-360)/2,360,360).data);t?o.textContent=t:window.requestAnimationFrame(u)}catch(e){console.error(e),window.requestAnimationFrame(u)}}else window.requestAnimationFrame(u)}t.srcObject=n,t.play(),i.addEventListener("click",()=>{o.textContent="",window.requestAnimationFrame(u)}),window.requestAnimationFrame(u)}catch(e){console.error(e)}})()}]);