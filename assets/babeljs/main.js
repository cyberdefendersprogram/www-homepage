// The following code is based off a toggle menu by @Bradcomp
// source: https://gist.github.com/Bradcomp/a9ef2ef322a8e8017443b626208999c1

document.addEventListener('DOMContentLoaded', () => {
  var burger = document.querySelector('.navbar-burger');
  var menu = document.querySelector('#'+burger.dataset.target);
  burger.addEventListener('click', function() {
      burger.classList.toggle('is-active');
      menu.classList.toggle('is-active');
  });

  var cancelDonateButton = document.getElementById('dismiss-ind-giving-2020');
  var cta = document.getElementById('cta-notification');
  cancelDonateButton.addEventListener('click', function(){
    cta.style.display = "none";
  })
});

