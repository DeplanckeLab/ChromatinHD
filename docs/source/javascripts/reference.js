document.querySelectorAll('.doc-attribute>:is(h1, h2, h3, h4, h5, h6)').forEach(function (el) {
    // add a tag to the attribute at the beginning
    var tag = document.createElement('span');
    tag.className = 'doc-attribute-tag';
    tag.textContent = "attribute"

    el.prepend(tag)

})

document.querySelectorAll('.doc-function>:is(h1, h2, h3, h4, h5, h6)').forEach(function (el) {
    // add a tag to the function at the beginning
    var tag = document.createElement('span');
    tag.className = 'doc-function-tag';
    // check if it's a method or a function
    console.log(el.closest('.doc-class'))
    if (el.closest('.doc-class')) {
        tag.textContent = "method"
    } else {
        tag.textContent = "function"
    }

    el.prepend(tag)

})


document.querySelectorAll('.doc-class>:is(h1, h2, h3, h4, h5, h6)').forEach(function (el) {
    // add a tag to the class at the beginning
    var tag = document.createElement('span');
    tag.className = 'doc-class-tag';
    tag.textContent = "class"

    el.prepend(tag)

})