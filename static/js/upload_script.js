// Hiển thị xem trước ảnh
$('#uploadImageInput').change(function (event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            $('.file-upload-content').show();
            $('.file-upload-image').attr('src', e.target.result);
        };
        reader.readAsDataURL(file);
    }
});

// Xóa ảnh đã tải
function removeUpload() {
    $('#uploadImageInput').val('');
    $('.file-upload-content').hide();
}

// Thêm hiệu ứng kéo thả (nếu cần thiết)
$('.file-upload').on('dragover', function () {
    $(this).addClass('file-dragging');
}).on('dragleave', function () {
    $(this).removeClass('file-dragging');
});
