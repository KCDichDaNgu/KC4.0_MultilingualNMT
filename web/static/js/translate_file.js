var ajax_transfile; // global variable for request transfile to server

init_source_tgt_lang();
function init_source_tgt_lang() {
  var source_lang = localStorage.getItem("transfile_src_lang")
    || $("#source_lang").children("option:selected").val();

  var target_lang = localStorage.getItem("transfile_tgt_lang")
    || $("#target_lang").children("option:selected").val();

  localStorage.setItem("transfile_src_lang", source_lang);
  localStorage.setItem("transfile_tgt_lang", target_lang);

  $("#source_lang option[value=" + source_lang + "]").prop('selected', true);
  $("#target_lang option[value=" + target_lang + "]").prop('selected', true);
}

/*Event for check file on Select files

args:
  type_alert: use alert option to notify error
              else use notify bar
*/
function isValid(type_alert = true) {
  var list_permit_type = ["xlsx", "docx", "pptx"];
  var file = $('input[type=file]');
  var file_extension = file.val().replace(/^.*\./, '');;
  if (!list_permit_type.includes(file_extension)) {
    if (type_alert) {
      alert("Error! Sorry, we only support files: .xlsx, .docx, .pptx, choose another file");
    } else {
      notify_err_msg("Error! Sorry, we only support files: .xlsx, .docx, .pptx, choose another file");
    }
    return false;
  }
  else {
    return true;
  }
}

/*event for select*/
$('#source_lang').on('change', function (e) {
  var new_src_lang = this.value;
  var old_src_lang = localStorage.getItem("transfile_src_lang");
  var old_tgt_lang = localStorage.getItem("transfile_tgt_lang");
  if (new_src_lang == old_tgt_lang) {
    // make a switch language
    $("#target_lang option[value=" + old_src_lang + "]").prop('selected', true);
    localStorage.setItem("transfile_tgt_lang", old_src_lang);
  }
  localStorage.setItem("transfile_src_lang", new_src_lang);
});

/*event for select*/
$('#target_lang').on('change', function (e) {
  var new_tgt_lang = this.value;
  var old_src_lang = localStorage.getItem("transfile_src_lang");
  var old_tgt_lang = localStorage.getItem("transfile_tgt_lang");
  if (new_tgt_lang == old_src_lang) {
    // make a switch language
    $("#source_lang option[value=" + old_tgt_lang + "]").prop('selected', true);
    localStorage.setItem("transfile_src_lang", old_tgt_lang);
  }
  localStorage.setItem("transfile_tgt_lang", new_tgt_lang);
});

/*event for switch language*/
$('#switch_lang').click(function () {
  var source_lang = localStorage.getItem("transfile_src_lang");
  var target_lang = localStorage.getItem("transfile_tgt_lang");
  // make a switch language
  $("#source_lang option[value=" + target_lang + "]").prop('selected', true);
  $("#target_lang option[value=" + source_lang + "]").prop('selected', true);
  localStorage.setItem("transfile_src_lang", target_lang);
  localStorage.setItem("transfile_tgt_lang", source_lang);
});

/*event for btn_abort*/
$('#btn_abort').click(function () {
  clearAllInterval();
  resetProgress();
  kill_ajax_transfile();
});

/*event for submit file*/
//global variables.
var local_progress_timer;
var target_estimate_timer;
var NUM_REPEAT_PROGRESS = 20;
$("form#form_transfile").submit(function (event) {
  event.preventDefault();

  if (!isValid(type_alert = false)) {
    return;
  }

  clearAllInterval(); //init to prevent unexpected cases
  resetProgress(); //init to prevent unexpected cases  

  $('#loader_bar_background').show();

  // show_loader_bar_background();

  var source_lang = $("#source_lang").children("option:selected").val();
  var target_lang = $("#target_lang").children("option:selected").val();

  var uploaded_file = $(this)[0];
  var formData = new FormData(uploaded_file);
  formData.append('json', JSON.stringify({ "direction": source_lang + target_lang }));

  //Time in milisecond.
  //local_progress_timer is a global variable
  console.log("run transfile")
  target_estimate_timer = setInterval(get_new_estimate_time, 5000);
  ajax_transfile = ajax_req_transfile($(this), formData);
});

/**
 * return ajax object
 */
var ajax_req_transfile = function (form_obj, formData) {
  return $.ajax({
    type: 'POST',
    url: form_obj.attr('action'),
    data: formData,
    cache: false,
    contentType: false,
    processData: false,
    success: function (response) {
      // console.log(response);      
      clearAllInterval();

      $("#percentage_value").text("99%");
      $("#percentage_loader_bar").width("39.6%");
      sleep(1500).then(() => {
        resetProgress();
        if (response) {
          // console.log("in response");            
          var result = response["data"];
          var data = result["data"];
          var status = result["status"];
          console.log(status);
          if (status) {
            window.open(data, '_blank');
            $("#download_file").attr("disabled", false);
            $("#download_file").attr("href", data);
          } else {
            notify_err_msg(data);
            $("#download_file").attr("disabled", true);
          }
        }
      });
    }

    // , timeout: 600000 //10'

    , error: function (err) {
      console.log("err");
      clearAllInterval();
      resetProgress();
      notify_err_msg("There is currently something wrong with file translating,\
                      please try later.");
    }
  });
}

// not use
// var get_id = function () {
//   var today = new Date();
//   var date = today.getFullYear() + '-' + (today.getMonth() + 1) + '-' + today.getDate();
//   var time
//     = today.getHours() + ":" + today.getMinutes()
//     + ":" + today.getSeconds() + ":" + today.getMilliseconds();
//   var rand_num = Math.floor(Math.random() * 100000);    
//   var _id = date + ' ' + time + ' ' + rand_num;
//   return _id;
// }

/*event before page refresh*/
window.onbeforeunload = function (event) {
  clearAllInterval();
  resetProgress();
};

var clearAllInterval = function () {
  clearInterval(local_progress_timer);
  clearInterval(target_estimate_timer);  
}

var resetProgress = function () {
  $('#loader_bar_background').hide();
  $("#percentage_value").text("0%");
  $("#percentage_loader_bar").width("0%");
  $("#percentage_value").attr("old_estimate_time", 0);
  $("#percentage_value").attr("old_interval", 500);   
}

/**
 * set to true
 * Args:
 *     + ajax_transfile: is global variable
 */
var kill_ajax_transfile = function () {
  if (ajax_transfile) {
    ajax_transfile.abort();
    display_notice(1, "Cancel translate file");
  }
}

/*event for download file*/
// $("#download_file").click(function () {
// });

/*update estimate time from server*/
var get_new_estimate_time = function () {
  //// TEMPORARY DISABLE GET NEW ESTIMATE TIME BY CALLING TO SERVER
  // $.ajax({
  //   type: 'POST',
  //   url: "/estimate_duration_translate_file",
  //   contentType: 'application/json;charset=UTF-8',
  //   dataType: "json",
  //   success: function (response) {
  //     if (response) {
  //       var result = response["data"];
  //       var new_estimate_time = result["estimate_time"];
  //       var old_estimate_time = parseFloat($("#percentage_value").attr("old_estimate_time"));
  //       if (old_estimate_time != new_estimate_time) {
  //         clearInterval(local_progress_timer);
  //         var old_interval = parseFloat($("#percentage_value").attr("old_interval") || "500");

  //         local_progress_timer = setInterval(update_progress_bar
  //           , old_interval, new_estimate_time);
  //       }
  //     }
  //   },
  //   error: function (err) {
  //     alert_error_ajax();
  //   }
  // });
  //// END

  // TEMPORARY USE ESTIMATE TIME IN LOCAL
  var new_estimate_time = 150000; //time in millisecond
  var old_estimate_time = parseFloat($("#percentage_value").attr("old_estimate_time"));
  console.log(new_estimate_time + " " +  old_estimate_time);
  if (old_estimate_time != new_estimate_time) {
    clearInterval(local_progress_timer);
    var old_interval = parseFloat($("#percentage_value").attr("old_interval") || "500");

    local_progress_timer = setInterval(update_progress_bar
      , old_interval, new_estimate_time);
  }
  //// END
}

/*function event for timeInterval*/
var update_progress_bar = function (new_estimate_time) {
  console.log("new_estimate_time:" + new_estimate_time);
  new_estimate_time = parseFloat(new_estimate_time);
  var old_estimate_time = parseFloat($("#percentage_value").attr("old_estimate_time"));
  var old_interval = parseFloat($("#percentage_value").attr("old_interval"));
  var old_percentage = get_percentage_number($("#percentage_value").text());
  //NUM_REPEAT_PROGRESS is a global variable
  // console.log("*:" + (old_percentage * old_estimate_time));
  var old_time = old_percentage > 0.0
    ? (old_percentage * old_estimate_time) : 0.0;
  console.log("old_percentage:" + old_percentage);
  console.log("old_estimate_time:" + old_estimate_time);
  console.log("old_interval:" + old_interval);
  var new_percentage;
  var new_percentage_text;
  var shouldRefreshInterval = false;
  if (new_estimate_time != old_estimate_time) {
    var new_interval = (1.0 * new_estimate_time - old_time) / NUM_REPEAT_PROGRESS;
    new_percentage = (old_time + new_interval) / new_estimate_time;
    // console.log("new_interval:" + new_interval);
    $("#percentage_value").attr("old_estimate_time", parseInt(new_estimate_time));
    $("#percentage_value").attr("old_interval", parseInt(new_interval));

    shouldRefreshInterval = true;
  }
  else {
    new_percentage = (old_time + old_interval) / old_estimate_time;
  }
  new_percentage_text = parseInt(new_percentage * 100) + "%";
  var new_percentage_loader_bar = parseInt(new_percentage * 0.4 * 100) + "%";
  if (new_percentage_text >= "93%") {
    new_percentage_text = "93%";
    new_percentage_loader_bar = "37.2%"
  }
  console.log("new_percentage:" + new_percentage);
  console.log("new_percentage_text:" + new_percentage_text);
  console.log("------------------------------");

  $("#percentage_value").text(new_percentage_text);
  $("#percentage_loader_bar").width(new_percentage_loader_bar);

  //maximum loader
  // console.log(get_percentage_number(new_percentage_loader_bar));
  if (get_percentage_number(new_percentage_loader_bar) >= 0.4) {
    // console.log(new_percentage);
    clearAllInterval();
  }

  if (shouldRefreshInterval) {
    clearInterval(local_progress_timer);
    local_progress_timer = setInterval(update_progress_bar
      , new_interval, new_estimate_time);
  }
}

/*get value from percentage_text
args: percentage_text: like 5%
return: value like 0.05.
*/
var get_percentage_number = function (percentage_text) {
  var value = parseFloat(percentage_text.substring(0, percentage_text.length - 1)) / 100;
  return value;
}

// var show_loader_bar_background = function () {
// }

/*nofity error based on local language of the browser*/
function notify_err_msg(msg = "There is no content in the file") {
  window.scrollTo(0, 0);

  var language = window.navigator.userLanguage || window.navigator.language;
  language = language.toLowerCase();
  // alert(language); //works IE/SAFARI/CHROME/FF  
  // if (language.indexOf("ja") != -1) {
  //   display_notice(0, "ファイルに内容がありません。!");
  // }
  // else if (language.indexOf("vi") != -1) {
  //   display_notice(0, "Không có nội dung trong tập tin!");
  // }
  // else {
  display_notice(0, msg);
  // }
}

function sleep(time) {
  return new Promise((resolve) => setTimeout(resolve, time));
}
//END TRANSLATE DOC
