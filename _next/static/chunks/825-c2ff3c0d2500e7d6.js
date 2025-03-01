"use strict";(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[825],{9205:function(e,t,n){n.d(t,{Z:function(){return a}});var r=n(2265);let o=e=>e.replace(/([a-z0-9])([A-Z])/g,"$1-$2").toLowerCase(),u=function(){for(var e=arguments.length,t=Array(e),n=0;n<e;n++)t[n]=arguments[n];return t.filter((e,t,n)=>!!e&&""!==e.trim()&&n.indexOf(e)===t).join(" ").trim()};var i={xmlns:"http://www.w3.org/2000/svg",width:24,height:24,viewBox:"0 0 24 24",fill:"none",stroke:"currentColor",strokeWidth:2,strokeLinecap:"round",strokeLinejoin:"round"};let l=(0,r.forwardRef)((e,t)=>{let{color:n="currentColor",size:o=24,strokeWidth:l=2,absoluteStrokeWidth:a,className:c="",children:f,iconNode:s,...d}=e;return(0,r.createElement)("svg",{ref:t,...i,width:o,height:o,stroke:n,strokeWidth:a?24*Number(l)/Number(o):l,className:u("lucide",c),...d},[...s.map(e=>{let[t,n]=e;return(0,r.createElement)(t,n)}),...Array.isArray(f)?f:[f]])}),a=(e,t)=>{let n=(0,r.forwardRef)((n,i)=>{let{className:a,...c}=n;return(0,r.createElement)(l,{ref:i,iconNode:t,className:u("lucide-".concat(o(e)),a),...c})});return n.displayName="".concat(e),n}},6741:function(e,t,n){n.d(t,{M:function(){return r}});function r(e,t,{checkForDefaultPrevented:n=!0}={}){return function(r){if(e?.(r),!1===n||!r.defaultPrevented)return t?.(r)}}},7822:function(e,t,n){n.d(t,{B:function(){return a}});var r=n(2265),o=n(3966),u=n(8575),i=n(7053),l=n(7437);function a(e){let t=e+"CollectionProvider",[n,a]=(0,o.b)(t),[c,f]=n(t,{collectionRef:{current:null},itemMap:new Map}),s=e=>{let{scope:t,children:n}=e,o=r.useRef(null),u=r.useRef(new Map).current;return(0,l.jsx)(c,{scope:t,itemMap:u,collectionRef:o,children:n})};s.displayName=t;let d=e+"CollectionSlot",m=r.forwardRef((e,t)=>{let{scope:n,children:r}=e,o=f(d,n),a=(0,u.e)(t,o.collectionRef);return(0,l.jsx)(i.g7,{ref:a,children:r})});m.displayName=d;let p=e+"CollectionItemSlot",v="data-radix-collection-item",g=r.forwardRef((e,t)=>{let{scope:n,children:o,...a}=e,c=r.useRef(null),s=(0,u.e)(t,c),d=f(p,n);return r.useEffect(()=>(d.itemMap.set(c,{ref:c,...a}),()=>void d.itemMap.delete(c))),(0,l.jsx)(i.g7,{[v]:"",ref:s,children:o})});return g.displayName=p,[{Provider:s,Slot:m,ItemSlot:g},function(t){let n=f(e+"CollectionConsumer",t);return r.useCallback(()=>{let e=n.collectionRef.current;if(!e)return[];let t=Array.from(e.querySelectorAll("[".concat(v,"]")));return Array.from(n.itemMap.values()).sort((e,n)=>t.indexOf(e.ref.current)-t.indexOf(n.ref.current))},[n.collectionRef,n.itemMap])},a]}},8575:function(e,t,n){n.d(t,{F:function(){return u},e:function(){return i}});var r=n(2265);function o(e,t){if("function"==typeof e)return e(t);null!=e&&(e.current=t)}function u(...e){return t=>{let n=!1,r=e.map(e=>{let r=o(e,t);return n||"function"!=typeof r||(n=!0),r});if(n)return()=>{for(let t=0;t<r.length;t++){let n=r[t];"function"==typeof n?n():o(e[t],null)}}}}function i(...e){return r.useCallback(u(...e),e)}},3966:function(e,t,n){n.d(t,{b:function(){return i},k:function(){return u}});var r=n(2265),o=n(7437);function u(e,t){let n=r.createContext(t),u=e=>{let{children:t,...u}=e,i=r.useMemo(()=>u,Object.values(u));return(0,o.jsx)(n.Provider,{value:i,children:t})};return u.displayName=e+"Provider",[u,function(o){let u=r.useContext(n);if(u)return u;if(void 0!==t)return t;throw Error(`\`${o}\` must be used within \`${e}\``)}]}function i(e,t=[]){let n=[],u=()=>{let t=n.map(e=>r.createContext(e));return function(n){let o=n?.[e]||t;return r.useMemo(()=>({[`__scope${e}`]:{...n,[e]:o}}),[n,o])}};return u.scopeName=e,[function(t,u){let i=r.createContext(u),l=n.length;n=[...n,u];let a=t=>{let{scope:n,children:u,...a}=t,c=n?.[e]?.[l]||i,f=r.useMemo(()=>a,Object.values(a));return(0,o.jsx)(c.Provider,{value:f,children:u})};return a.displayName=t+"Provider",[a,function(n,o){let a=o?.[e]?.[l]||i,c=r.useContext(a);if(c)return c;if(void 0!==u)return u;throw Error(`\`${n}\` must be used within \`${t}\``)}]},function(...e){let t=e[0];if(1===e.length)return t;let n=()=>{let n=e.map(e=>({useScope:e(),scopeName:e.scopeName}));return function(e){let o=n.reduce((t,{useScope:n,scopeName:r})=>{let o=n(e)[`__scope${r}`];return{...t,...o}},{});return r.useMemo(()=>({[`__scope${t.scopeName}`]:o}),[o])}};return n.scopeName=t.scopeName,n}(u,...t)]}},9114:function(e,t,n){n.d(t,{gm:function(){return u}});var r=n(2265);n(7437);var o=r.createContext(void 0);function u(e){let t=r.useContext(o);return e||t||"ltr"}},9255:function(e,t,n){n.d(t,{M:function(){return a}});var r,o=n(2265),u=n(1188),i=(r||(r=n.t(o,2)))["useId".toString()]||(()=>void 0),l=0;function a(e){let[t,n]=o.useState(i());return(0,u.b)(()=>{e||n(e=>e??String(l++))},[e]),e||(t?`radix-${t}`:"")}},1599:function(e,t,n){n.d(t,{z:function(){return i}});var r=n(2265),o=n(8575),u=n(1188),i=e=>{var t,n;let i,a;let{present:c,children:f}=e,s=function(e){var t,n;let[o,i]=r.useState(),a=r.useRef({}),c=r.useRef(e),f=r.useRef("none"),[s,d]=(t=e?"mounted":"unmounted",n={mounted:{UNMOUNT:"unmounted",ANIMATION_OUT:"unmountSuspended"},unmountSuspended:{MOUNT:"mounted",ANIMATION_END:"unmounted"},unmounted:{MOUNT:"mounted"}},r.useReducer((e,t)=>{let r=n[e][t];return null!=r?r:e},t));return r.useEffect(()=>{let e=l(a.current);f.current="mounted"===s?e:"none"},[s]),(0,u.b)(()=>{let t=a.current,n=c.current;if(n!==e){let r=f.current,o=l(t);e?d("MOUNT"):"none"===o||(null==t?void 0:t.display)==="none"?d("UNMOUNT"):n&&r!==o?d("ANIMATION_OUT"):d("UNMOUNT"),c.current=e}},[e,d]),(0,u.b)(()=>{if(o){var e;let t;let n=null!==(e=o.ownerDocument.defaultView)&&void 0!==e?e:window,r=e=>{let r=l(a.current).includes(e.animationName);if(e.target===o&&r&&(d("ANIMATION_END"),!c.current)){let e=o.style.animationFillMode;o.style.animationFillMode="forwards",t=n.setTimeout(()=>{"forwards"===o.style.animationFillMode&&(o.style.animationFillMode=e)})}},u=e=>{e.target===o&&(f.current=l(a.current))};return o.addEventListener("animationstart",u),o.addEventListener("animationcancel",r),o.addEventListener("animationend",r),()=>{n.clearTimeout(t),o.removeEventListener("animationstart",u),o.removeEventListener("animationcancel",r),o.removeEventListener("animationend",r)}}d("ANIMATION_END")},[o,d]),{isPresent:["mounted","unmountSuspended"].includes(s),ref:r.useCallback(e=>{e&&(a.current=getComputedStyle(e)),i(e)},[])}}(c),d="function"==typeof f?f({present:s.isPresent}):r.Children.only(f),m=(0,o.e)(s.ref,(i=null===(t=Object.getOwnPropertyDescriptor(d.props,"ref"))||void 0===t?void 0:t.get)&&"isReactWarning"in i&&i.isReactWarning?d.ref:(i=null===(n=Object.getOwnPropertyDescriptor(d,"ref"))||void 0===n?void 0:n.get)&&"isReactWarning"in i&&i.isReactWarning?d.props.ref:d.props.ref||d.ref);return"function"==typeof f||s.isPresent?r.cloneElement(d,{ref:m}):null};function l(e){return(null==e?void 0:e.animationName)||"none"}i.displayName="Presence"},6840:function(e,t,n){n.d(t,{WV:function(){return l},jH:function(){return a}});var r=n(2265),o=n(4887),u=n(7053),i=n(7437),l=["a","button","div","form","h2","h3","img","input","label","li","nav","ol","p","span","svg","ul"].reduce((e,t)=>{let n=r.forwardRef((e,n)=>{let{asChild:r,...o}=e,l=r?u.g7:t;return"undefined"!=typeof window&&(window[Symbol.for("radix-ui")]=!0),(0,i.jsx)(l,{...o,ref:n})});return n.displayName=`Primitive.${t}`,{...e,[t]:n}},{});function a(e,t){e&&o.flushSync(()=>e.dispatchEvent(t))}},1353:function(e,t,n){n.d(t,{Pc:function(){return b},ck:function(){return j},fC:function(){return S}});var r=n(2265),o=n(6741),u=n(7822),i=n(8575),l=n(3966),a=n(9255),c=n(6840),f=n(6606),s=n(886),d=n(9114),m=n(7437),p="rovingFocusGroup.onEntryFocus",v={bubbles:!1,cancelable:!0},g="RovingFocusGroup",[w,y,h]=(0,u.B)(g),[N,b]=(0,l.b)(g,[h]),[R,M]=N(g),E=r.forwardRef((e,t)=>(0,m.jsx)(w.Provider,{scope:e.__scopeRovingFocusGroup,children:(0,m.jsx)(w.Slot,{scope:e.__scopeRovingFocusGroup,children:(0,m.jsx)(x,{...e,ref:t})})}));E.displayName=g;var x=r.forwardRef((e,t)=>{let{__scopeRovingFocusGroup:n,orientation:u,loop:l=!1,dir:a,currentTabStopId:g,defaultCurrentTabStopId:w,onCurrentTabStopIdChange:h,onEntryFocus:N,preventScrollOnEntryFocus:b=!1,...M}=e,E=r.useRef(null),x=(0,i.e)(t,E),C=(0,d.gm)(a),[A=null,T]=(0,s.T)({prop:g,defaultProp:w,onChange:h}),[S,j]=r.useState(!1),I=(0,f.W)(N),F=y(n),k=r.useRef(!1),[P,D]=r.useState(0);return r.useEffect(()=>{let e=E.current;if(e)return e.addEventListener(p,I),()=>e.removeEventListener(p,I)},[I]),(0,m.jsx)(R,{scope:n,orientation:u,dir:C,loop:l,currentTabStopId:A,onItemFocus:r.useCallback(e=>T(e),[T]),onItemShiftTab:r.useCallback(()=>j(!0),[]),onFocusableItemAdd:r.useCallback(()=>D(e=>e+1),[]),onFocusableItemRemove:r.useCallback(()=>D(e=>e-1),[]),children:(0,m.jsx)(c.WV.div,{tabIndex:S||0===P?-1:0,"data-orientation":u,...M,ref:x,style:{outline:"none",...e.style},onMouseDown:(0,o.M)(e.onMouseDown,()=>{k.current=!0}),onFocus:(0,o.M)(e.onFocus,e=>{let t=!k.current;if(e.target===e.currentTarget&&t&&!S){let t=new CustomEvent(p,v);if(e.currentTarget.dispatchEvent(t),!t.defaultPrevented){let e=F().filter(e=>e.focusable);O([e.find(e=>e.active),e.find(e=>e.id===A),...e].filter(Boolean).map(e=>e.ref.current),b)}}k.current=!1}),onBlur:(0,o.M)(e.onBlur,()=>j(!1))})})}),C="RovingFocusGroupItem",A=r.forwardRef((e,t)=>{let{__scopeRovingFocusGroup:n,focusable:u=!0,active:i=!1,tabStopId:l,...f}=e,s=(0,a.M)(),d=l||s,p=M(C,n),v=p.currentTabStopId===d,g=y(n),{onFocusableItemAdd:h,onFocusableItemRemove:N}=p;return r.useEffect(()=>{if(u)return h(),()=>N()},[u,h,N]),(0,m.jsx)(w.ItemSlot,{scope:n,id:d,focusable:u,active:i,children:(0,m.jsx)(c.WV.span,{tabIndex:v?0:-1,"data-orientation":p.orientation,...f,ref:t,onMouseDown:(0,o.M)(e.onMouseDown,e=>{u?p.onItemFocus(d):e.preventDefault()}),onFocus:(0,o.M)(e.onFocus,()=>p.onItemFocus(d)),onKeyDown:(0,o.M)(e.onKeyDown,e=>{if("Tab"===e.key&&e.shiftKey){p.onItemShiftTab();return}if(e.target!==e.currentTarget)return;let t=function(e,t,n){var r;let o=(r=e.key,"rtl"!==n?r:"ArrowLeft"===r?"ArrowRight":"ArrowRight"===r?"ArrowLeft":r);if(!("vertical"===t&&["ArrowLeft","ArrowRight"].includes(o))&&!("horizontal"===t&&["ArrowUp","ArrowDown"].includes(o)))return T[o]}(e,p.orientation,p.dir);if(void 0!==t){if(e.metaKey||e.ctrlKey||e.altKey||e.shiftKey)return;e.preventDefault();let o=g().filter(e=>e.focusable).map(e=>e.ref.current);if("last"===t)o.reverse();else if("prev"===t||"next"===t){var n,r;"prev"===t&&o.reverse();let u=o.indexOf(e.currentTarget);o=p.loop?(n=o,r=u+1,n.map((e,t)=>n[(r+t)%n.length])):o.slice(u+1)}setTimeout(()=>O(o))}})})})});A.displayName=C;var T={ArrowLeft:"prev",ArrowUp:"prev",ArrowRight:"next",ArrowDown:"next",PageUp:"first",Home:"first",PageDown:"last",End:"last"};function O(e){let t=arguments.length>1&&void 0!==arguments[1]&&arguments[1],n=document.activeElement;for(let r of e)if(r===n||(r.focus({preventScroll:t}),document.activeElement!==n))return}var S=E,j=A},7053:function(e,t,n){n.d(t,{g7:function(){return i}});var r=n(2265),o=n(8575),u=n(7437),i=r.forwardRef((e,t)=>{let{children:n,...o}=e,i=r.Children.toArray(n),a=i.find(c);if(a){let e=a.props.children,n=i.map(t=>t!==a?t:r.Children.count(e)>1?r.Children.only(null):r.isValidElement(e)?e.props.children:null);return(0,u.jsx)(l,{...o,ref:t,children:r.isValidElement(e)?r.cloneElement(e,void 0,n):null})}return(0,u.jsx)(l,{...o,ref:t,children:n})});i.displayName="Slot";var l=r.forwardRef((e,t)=>{let{children:n,...u}=e;if(r.isValidElement(n)){let e,i;let l=(e=Object.getOwnPropertyDescriptor(n.props,"ref")?.get)&&"isReactWarning"in e&&e.isReactWarning?n.ref:(e=Object.getOwnPropertyDescriptor(n,"ref")?.get)&&"isReactWarning"in e&&e.isReactWarning?n.props.ref:n.props.ref||n.ref,a=function(e,t){let n={...t};for(let r in t){let o=e[r],u=t[r];/^on[A-Z]/.test(r)?o&&u?n[r]=(...e)=>{u(...e),o(...e)}:o&&(n[r]=o):"style"===r?n[r]={...o,...u}:"className"===r&&(n[r]=[o,u].filter(Boolean).join(" "))}return{...e,...n}}(u,n.props);return n.type!==r.Fragment&&(a.ref=t?(0,o.F)(t,l):l),r.cloneElement(n,a)}return r.Children.count(n)>1?r.Children.only(null):null});l.displayName="SlotClone";var a=({children:e})=>(0,u.jsx)(u.Fragment,{children:e});function c(e){return r.isValidElement(e)&&e.type===a}},6606:function(e,t,n){n.d(t,{W:function(){return o}});var r=n(2265);function o(e){let t=r.useRef(e);return r.useEffect(()=>{t.current=e}),r.useMemo(()=>(...e)=>t.current?.(...e),[])}},886:function(e,t,n){n.d(t,{T:function(){return u}});var r=n(2265),o=n(6606);function u({prop:e,defaultProp:t,onChange:n=()=>{}}){let[u,i]=function({defaultProp:e,onChange:t}){let n=r.useState(e),[u]=n,i=r.useRef(u),l=(0,o.W)(t);return r.useEffect(()=>{i.current!==u&&(l(u),i.current=u)},[u,i,l]),n}({defaultProp:t,onChange:n}),l=void 0!==e,a=l?e:u,c=(0,o.W)(n);return[a,r.useCallback(t=>{if(l){let n="function"==typeof t?t(e):t;n!==e&&c(n)}else i(t)},[l,e,i,c])]}},1188:function(e,t,n){n.d(t,{b:function(){return o}});var r=n(2265),o=globalThis?.document?r.useLayoutEffect:()=>{}}}]);