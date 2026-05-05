document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector(".needs-soft-check");
    if (!form) {
        return;
    }

    form.addEventListener("submit", (event) => {
        const requiredFields = form.querySelectorAll("[required]");
        let firstInvalid = null;

        requiredFields.forEach((field) => {
            const isEmpty = !String(field.value || "").trim();
            field.classList.toggle("is-invalid", isEmpty);
            if (isEmpty && !firstInvalid) {
                firstInvalid = field;
            }
        });

        if (firstInvalid) {
            event.preventDefault();
            firstInvalid.focus({ preventScroll: true });
            firstInvalid.scrollIntoView({ behavior: "smooth", block: "center" });
        }
    });
});
