/*event for login*/
$(".get_password_form").submit(function (e) {
    e.preventDefault();
}).validate({
    rules: {
        email: {
            required: true,
            email: true
        },
    },
    messages: {
        email: {
            required: "You must enter an email address.",
            email: "You've entered an invalid email address."
        },
    },
    submitHandler: function (form) {
        var email = $('input[name="email"]').val();

        $('#loader').show();
        data = {
            "data": {
                "email": email,
            }
        }
        $.ajax({
            type: "POST",
            url: "/get_password",
            data: JSON.stringify(data),
            contentType: 'application/json;charset=UTF-8',
            dataType: "json",
            success: function (response) {
                $('#loader').hide();
                result = response["data"];
                display_notice(result["success"], result["msg"]);
            },
            error: function (err) {
                $('#loader').hide();
                alert_error_ajax();
            }
        });
    }
});

///START
var key_enter_evt = function (event) {
    if (event.keyCode === 13) {
        event.preventDefault();
        document.getElementById("btn_forgot_password").click();
    }
}

var input = document.getElementById("email");
input.addEventListener("keyup", key_enter_evt);
///END