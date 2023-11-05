function toc_animate(){
    md_toc = document.querySelectorAll('[id^="markdown-toc-"]')
    if (md_toc.length!=0){
        var cur_active = md_toc[0]
        for (let i = 0; i < md_toc.length; i++) {
            md_toc[i].addEventListener('click', () => {
                cur_active.setAttribute('aria-current', false)
                md_toc[i].setAttribute('aria-current', true)
                cur_active  = md_toc[i]
            });
        };
    };
}

document.addEventListener("DOMContentLoaded", () => {
    toc_animate()
});