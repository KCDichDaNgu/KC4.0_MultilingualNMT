/*event for login*/
$("#signin_form").submit(function (e) {
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
    submitHandler: function (form) {
        var email = $('input[name="email"]').val();
        var password = $('input[name="password"]').val();

        $('#loader').show();
        data = {
            "data": {
                "email": email,
                "password": password
            }
        }
        $.ajax({
            type: "POST",
            url: "/login_exe",
            data: JSON.stringify(data),
            contentType: 'application/json;charset=UTF-8',
            dataType: "json",
            success: function (response) {
                $('#loader').hide();
                result = response["data"];
                if (!result["success"]) {                    
                    display_notice(result["success"], result["msg"]);
                }
                else {
                    window.location.href = result["msg"];
                }
            },
            error: function (err) {
                $('#loader').hide();
                alert_error_ajax();
            }
        });
    },
});

///START
var key_enter_evt = function (event) {
    if (event.keyCode === 13) {
        event.preventDefault();
        document.getElementById("btn_login").click();
    }
}

var input = document.getElementById("email");
input.addEventListener("keyup", key_enter_evt);
var input = document.getElementById("password");
input.addEventListener("keyup", key_enter_evt);
///END