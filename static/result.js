document.addEventListener("DOMContentLoaded", set_bar_widths)

function set_bar_widths() {
    const label_bars = document.getElementsByClassName("label-bar")
    let num_labels = 0
    for (const label_bar of label_bars) {
        num_labels++;
        const percentage_str = label_bar.getElementsByClassName("num").item(0).innerHTML
        const id_name = "label-bar-num-" + num_labels.toString()
        label_bar.getElementsByClassName("label-percentage").item(0).id = id_name
        document.getElementById(id_name).style.width = percentage_str
    }
}