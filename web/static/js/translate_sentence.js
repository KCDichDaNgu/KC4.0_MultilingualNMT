var IS_LIMIT_SENTENCE_LENGTH = false;
var MAX_LENGTH_WORD = 500;
var END_SENTENCE_SIGNS = ".!?;\n";

init_source_target_lang();

/*init temporary storage for source and taget language*/
function init_source_target_lang() {
  var source_lang = localStorage.getItem("transSent_src_lang")
    || $("#source_lang").children("option:selected").val();

  var target_lang = localStorage.getItem("transSent_tgt_lang")
    || $("#target_lang").children("option:selected").val();

  localStorage.setItem("transSent_src_lang", source_lang);
  localStorage.setItem("transSent_tgt_lang", target_lang);

  $("#source_lang option[value=" + source_lang + "]").prop('selected', true);
  $("#target_lang option[value=" + target_lang + "]").prop('selected', true);
}

/*event for select language*/
$('#source_lang').on('change', function (e) {
  var new_source_lang = this.value;
  var old_source_lang = localStorage.getItem("transSent_src_lang");
  var old_target_lang = localStorage.getItem("transSent_tgt_lang");
  if (new_source_lang == old_target_lang) {
    // make a switch language
    $("#target_lang option[value=" + old_source_lang + "]").prop('selected', true);
    localStorage.setItem("transSent_tgt_lang", old_source_lang);
  }
  localStorage.setItem("transSent_src_lang", new_source_lang);
});

/*event for select language*/
$('#target_lang').on('change', function (e) {
  var new_target_lang = this.value;
  var old_source_lang = localStorage.getItem("transSent_src_lang");
  var old_target_lang = localStorage.getItem("transSent_tgt_lang");
  if (new_target_lang == old_source_lang) {
    // make a switch language
    $("#source_lang option[value=" + old_target_lang + "]").prop('selected', true);
    localStorage.setItem("transSent_src_lang", old_target_lang);
  }
  localStorage.setItem("transSent_tgt_lang", new_target_lang);
});

/*event for switch between two languages*/
$('#switch_lang').click(function () {
  var tmp = $('textarea#text_target').val();
  $('textarea#text_target').val($('textarea#text_source').val());
  $('textarea#text_source').val(tmp);
    
  var source_lang = localStorage.getItem("transSent_src_lang");
  var target_lang = localStorage.getItem("transSent_tgt_lang");
  // make a switch language
  $("#source_lang option[value=" + target_lang + "]").prop('selected', true);
  $("#target_lang option[value=" + source_lang + "]").prop('selected', true);
  localStorage.setItem("transSent_src_lang", target_lang);
  localStorage.setItem("transSent_tgt_lang", source_lang);
});

/*event for clear input textarea*/
$('#btn_clear_input').click(function () {
  $('textarea#text_source').val("");
  $('textarea#text_target').val("");
})

/*event for translate sentence*/
function translateFunction() {
  // $('#loader').show();  
  var source_lang = $("#source_lang").children("option:selected").val();
  var target_lang = $("#target_lang").children("option:selected").val();
  var source_paratext = $('textarea#text_source').val();

  if (!isValidInputText(source_paratext)) {
    return;
  }

  $("textarea#text_target").css({ 'color': 'black' });
  $("textarea#text_target").val("translating...")
  // var source_sentences = splitIntoSentence(source_paratext);
  var data = {
    "data": source_paratext
    , "direction": source_lang + target_lang
  };

  console.log(JSON.stringify(data));
  $.ajax({
    type: "POST",
    url: "/translate_paragraphs",
    data: JSON.stringify(data),
    contentType: "application/json; charset=utf-8",
    dataType: "json"
    , success: function (response) {
      //$('#loader').hide();
      // console.log(result);
      var result = response["data"];
      var res_status = result["status"];

      if (!res_status) {
        var target_param = result["data"];
        notify_err_msg(target_param);
        $("textarea#text_target").val(target_param)
        $("textarea#text_target").css({ 'color': 'red' });
        return;
      } else {
        $("textarea#text_target").css({ 'color': 'black' });
        var target_param = result["data"];
        $("textarea#text_target").val(target_param)
      }
    }

    // , timeout: 600000 //10'

    , error: function (err) {
      $("textarea#text_target").val("")
      alert_error_ajax();
    }
  });
}

function oninput_text_func() {
  var source_paratext = $('textarea#text_source').val();
  // var target_paratext = $('textarea#text_target').val();  
  if (!source_paratext.trim()) {
    $("textarea#text_target").val("")
  }
  else if (sessionStorage.getItem("is_cut_paste") == "true") {
    translateFunction()
  }
  // Final step set to false for is_cut_paste
  sessionStorage.setItem("is_cut_paste", "false")
}

function oncut_textare_func() {
  sessionStorage.setItem("is_cut_paste", "true")
}

function onpaste_textare_func() {
  sessionStorage.setItem("is_cut_paste", "true")
}

/*nofity error based on local language of the browser*/
function notify_err_msg(msg = "There is no content to translate!") {
  window.scrollTo(0, 0);

  var language = window.navigator.userLanguage || window.navigator.language;
  language = language.toLowerCase();
  // alert(language); //works IE/SAFARI/CHROME/FF

  // if (language.indexOf("ja") != -1) {
  //   display_notice(0, "翻訳する内容はありません。!");
  // }
  // else if (language.indexOf("vi") != -1) {
  //   display_notice(0, "Không có nội dung để dịch!");
  // }
  // else {
  display_notice(0, msg);
  // }
}

/*nofity error based on local language of the browser*/
function notify_sentence_to_long() {
  window.scrollTo(0, 0);

  var language = window.navigator.userLanguage || window.navigator.language;
  language = language.toLowerCase();
  // alert(language); //works IE/SAFARI/CHROME/FF

  // if (language.indexOf("ja") != -1) {
  //   display_notice(0, "文が長すぎます。!");
  // }
  // else if (language.indexOf("vi") != -1) {
  //   display_notice(0, "Câu quá dài!");
  // }
  // else {
  display_notice(0, "The sentence is too long!");
  // }
}

/*direction event of languages */
function changeDirection() {
  var direction = document.getElementById("direction");
  if (direction.value == "envi") {
    direction.value = "vien";
    direction.innerHTML = "Vietnamese - English";
  } else {
    direction.value = "envi";
    direction.innerHTML = "English - Vietnamese";
  }
  $('textarea#text_source').val("");
  $("textarea#text_target").val("")
}

/*split an array long text to an array of smaller text

args: 
  paratext: an array of sentences containing long ones.

return:
  sentences: an array of short sentences.
*/
function splitIntoSentence(paratext) {
  paratext = paratext.trim();
  var start = -1
  var sentences = [];
  for (var i = 0; i < paratext.length; i++) {
    if (start == -1)
      start = i
    if ((END_SENTENCE_SIGNS.includes(paratext[i])) //avoid float number
      || i == paratext.length - 1) {

      sentences.push(paratext.substring(start, i + 1));
      start = -1
    }
  }
  // console.log(paratext)
  console.log(sentences)
  return sentences
}

/*build a text of an array of target language sentences 
based on an array of source language sentences 

args: 
  tgt_sentences: an array contains target language sentences. 
  src_sentences: an array contains source language sentences.

return:
  sentences: a text containing target language sentences.
*/
function joinTargetSenteces(tgt_sentences, src_sentences) {
  var tgt_paratext = "";
  for (var i = 0; i < src_sentences.length; i++) {
    var src = src_sentences[i];
    var tgt = tgt_sentences[i];
    var src_ending = src.trim()[src.length - 1];
    var tgt_ending = tgt.trim()[tgt.length - 1];
    if (END_SENTENCE_SIGNS.includes(src_ending)
      && src_ending != tgt_ending) {

      tgt += src_ending + " ";
    }

    tgt_paratext += tgt + " ";
  }

  return tgt_paratext;
}

function isValidInputText(source_paratext) {

  if (!source_paratext) {
    notify_err_msg();
    return false;
  }

  if (IS_LIMIT_SENTENCE_LENGTH
    && source_paratext.trim().split(" ").length > MAX_LENGTH_WORD) {
    notify_sentence_to_long();
    return false;
  }


  return true;
}

var shortcut_translate_evt = function (event) {
  if (event.keyCode == 13 && event.ctrlKey) {
    // alert("Ctrl+enter");
    event.preventDefault();
    document.getElementById("btn_translate").click();
  }
}

var text_source = document.getElementById("text_source");
text_source.addEventListener("keydown", shortcut_translate_evt);








