/*Display a notification on top of screen.

args:
    notice_type: 0 for error and 1 for success.
    content: the text to display.
    time_display: in millisecond
*/
var display_notice = function (notice_type, content, time_display=5000) {
  // var modal = document.getElementById('myModal');
  if (notice_type) {
    myAlert.style.backgroundColor = "#2196F3";
  } else {
    myAlert.style.backgroundColor = "#f44336";
  }
  var notice_content = document.getElementById("notice_content");
  notice_content.innerHTML = content;//use <br> to break line in text
  myAlert.style.display = "block";
  myAlert.style.width = "100%";
  setTimeout(function () {
    myAlert.style.display = "none";
  }, time_display);

  $('#closeAlertBtn').click(function () {
    var myAlert = document.getElementById('myAlert');
    myAlert.style.display = "none";
    // myAlert.style.backgroundColor = "#2196F3";    
  })
}

function alert_error_ajax() {
  var language = window.navigator.userLanguage || window.navigator.language;
  language = language.toLowerCase();
  // alert(language); //works IE/SAFARI/CHROME/FF

  if (language.indexOf("ja") != -1) {
    display_notice(0, "何かがおかしい！"
      + "\"インターネット接続を確認してください "
      + " \もう一度やり直してください。");
  }
  else if (language.indexOf("vi") != -1) {
    alert("Có gì đó không đúng!"
      + "\ Vui lòng kiểm tra kết nối Internet của bạn"
      + "\ Và thử lại.");
  }
  else {
    display_notice(0, "There is something wrong!"
      + "\nPlease checkout your Internet connection"
      + "\nAnd try again.");
  }
}