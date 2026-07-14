// Toggle a long retrieved-document block between clamped and full height.
function toggleMore(b) {
  var d = b.parentNode;
  var open = d.classList.toggle('open');
  b.textContent = open ? 'Show less' : 'Read more';
}
