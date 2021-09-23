/*event for add user*/
$(".signup").submit(function (e) {
  e.preventDefault();
}).validate({
  rules: {
    email: {
      required: true,
      email: true
    },
    password: {
      required: true,
      minlength: 6
    }
  },
  messages: {
    email: {
      required: "You must enter an email address.",
      email: "You've entered an invalid email address."
    },
    password: {
      required: "You must enter a password.",
      minlength: "Your password must be at least {0} characters."
    }
  },
  submitHandler: function(form) { 
    var email = $('input[name="email"]').val();
    var password = $('input[name="password"]').val();
    var username = $('input[name="username"]').val();

    $('#loader').show();

    data = {
      "data": {
        "email": email,
        "username": username,
        "password": password,
        "role": "user"
      }
    }
    // console.log(JSON.stringify(data));
    $.ajax({
      type: "POST",
      url: "/add_user_exe",
      data: JSON.stringify(data),
      contentType: 'application/json;charset=UTF-8',
      dataType: "json",
      success: function (response) {
        result = response["data"];
        $('#loader').hide();
        console.log(result);
        display_notice(result["success"], result["msg"]);
      },
      error: function (err) {
        $('#loader').hide();
        alert_error_ajax();
      }
    });    
  }
});
